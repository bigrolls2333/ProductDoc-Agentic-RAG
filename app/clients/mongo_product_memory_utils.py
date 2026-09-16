"""MongoDB persistence helpers for product-level long-term memory."""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

# Bound the number of retained query patterns per product.
MAX_COMMON_QUESTIONS = 20


class ProductMemoryMongoTool:
    """Manage the product-memory collection and its indexes."""

    def __init__(self):
        """Initialize the MongoDB client, collection, and indexes."""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db["product_memory"]

            self.collection.create_index([("canonical_name", ASCENDING)], unique=True)
            self.collection.create_index([("aliases", ASCENDING)])
            self.collection.create_index([("updated_at", ASCENDING)])

            logging.info(f"Product memory connected: {self.db_name}.product_memory")
        except Exception as e:
            logging.error(f"Product memory initialization failed: {e}")
            raise


_product_memory_mongo_tool: Optional[ProductMemoryMongoTool] = None

try:
    _product_memory_mongo_tool = ProductMemoryMongoTool()
except Exception as e:
    logging.warning(f"ProductMemoryMongoTool module initialization failed: {e}")


def get_product_memory_mongo_tool() -> Optional[ProductMemoryMongoTool]:
    """Return the lazily initialized ProductMemoryMongoTool singleton."""
    global _product_memory_mongo_tool
    if _product_memory_mongo_tool is None:
        try:
            _product_memory_mongo_tool = ProductMemoryMongoTool()
        except Exception as e:
            logging.warning(f"Lazy initialization of ProductMemoryMongoTool failed: {e}")
            return None
    return _product_memory_mongo_tool


def get_product_memory(canonical_name: str) -> dict | None:
    """Return product memory for a canonical product name."""
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
        logging.warning(f"Failed to read product_memory: canonical_name={canonical_name}, error={e}")
        return None


def upsert_product_memory(
    canonical_name: str,
    query_pattern: str,
    answer_summary: str,
    evidence_chunk_ids: list,
    route_meta: dict,
) -> bool:
    """Upsert a bounded common-question record for a product."""
    if not canonical_name or not query_pattern:
        return False

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return False

    try:
        now = time.time()
        summary = (answer_summary or "")[:500]

        existing = tool.collection.find_one({"canonical_name": canonical_name})
        if existing is None:
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

        common_questions = existing.get("common_questions") or []
        found = False
        for cq in common_questions:
            existing_pattern = cq.get("query_pattern", "")
            if _patterns_similar(query_pattern, existing_pattern):
                cq["frequency"] = cq.get("frequency", 0) + 1
                cq["last_asked_at"] = now
                cq["answer_summary"] = summary
                cq["route_meta"] = route_meta or {}
                found = True
                break

        if not found:
            common_questions.append({
                "query_pattern": query_pattern,
                "answer_summary": summary,
                "evidence_chunk_ids": evidence_chunk_ids or [],
                "frequency": 1,
                "last_asked_at": now,
                "route_meta": route_meta or {},
            })
            if len(common_questions) > MAX_COMMON_QUESTIONS:
                common_questions.sort(key=lambda x: x.get("last_asked_at", 0))
                common_questions = common_questions[-MAX_COMMON_QUESTIONS:]

        tool.collection.update_one(
            {"canonical_name": canonical_name},
            {"$set": {
                "common_questions": common_questions,
                "updated_at": now,
            }}
        )
        return True

    except PyMongoError as e:
        logging.warning(f"Failed to write product_memory: canonical_name={canonical_name}, error={e}")
        return False


def _patterns_similar(pattern1: str, pattern2: str) -> bool:
    """Compare normalized query patterns using equality or containment."""
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
    """Find product memories by canonical name or alias."""
    if not query:
        return []

    tool = get_product_memory_mongo_tool()
    if tool is None:
        return []

    try:
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
        logging.warning(f"Failed to search product_memory: query={query}, error={e}")
        return []


def add_product_alias(canonical_name: str, alias: str) -> bool:
    """Add an alias to a product memory document."""
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
        logging.warning(f"Failed to add product alias: canonical_name={canonical_name}, error={e}")
        return False


def link_doc_to_product(canonical_name: str, doc_id: str) -> bool:
    """Associate a source document with a product memory document."""
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
        logging.warning(f"Failed to link document to product: canonical_name={canonical_name}, error={e}")
        return False
