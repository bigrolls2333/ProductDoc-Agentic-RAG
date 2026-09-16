"""MongoDB-backed session memory for conversational query context."""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()


class SessionMemoryMongoTool:
    """Manage the session-memory collection and its indexes."""

    def __init__(self):
        """Initialize the MongoDB client, collection, and indexes."""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.collection = self.db["session_memory"]

            self.collection.create_index([("session_id", ASCENDING)], unique=True)
            self.collection.create_index([("updated_at", ASCENDING)])

            logging.info(f"Session memory connected: {self.db_name}.session_memory")
        except Exception as e:
            logging.error(f"Session memory initialization failed: {e}")
            raise


_session_memory_mongo_tool: Optional[SessionMemoryMongoTool] = None

try:
    _session_memory_mongo_tool = SessionMemoryMongoTool()
except Exception as e:
    logging.warning(f"SessionMemoryMongoTool module initialization failed: {e}")


def get_session_memory_mongo_tool() -> Optional[SessionMemoryMongoTool]:
    """Return the lazily initialized SessionMemoryMongoTool singleton."""
    global _session_memory_mongo_tool
    if _session_memory_mongo_tool is None:
        try:
            _session_memory_mongo_tool = SessionMemoryMongoTool()
        except Exception as e:
            logging.warning(f"Lazy initialization of SessionMemoryMongoTool failed: {e}")
            return None
    return _session_memory_mongo_tool


def get_session_memory(session_id: str) -> dict:
    """Return session memory or an empty mapping when unavailable."""
    if not session_id:
        return {}

    tool = get_session_memory_mongo_tool()
    if tool is None:
        return {}

    try:
        doc = tool.collection.find_one({"session_id": session_id})
        if doc is None:
            return {}
        doc.pop("_id", None)
        return doc
    except PyMongoError as e:
        logging.warning(f"Session memory read failed; returning empty state: session_id={session_id}, error={e}")
        return {}


def upsert_session_memory(session_id: str, updates: dict) -> bool:
    """Merge fields into a session-memory document."""
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
        logging.warning(f"Session memory write failed: session_id={session_id}, error={e}")
        return False


def clear_session_memory(session_id: str) -> int:
    """Delete the memory document for a session and return the delete count."""
    if not session_id:
        return 0

    tool = get_session_memory_mongo_tool()
    if tool is None:
        return 0

    try:
        result = tool.collection.delete_one({"session_id": session_id})
        return result.deleted_count
    except PyMongoError as e:
        logging.warning(f"Session memory delete failed: session_id={session_id}, error={e}")
        return 0


def clear_pending_clarification(session_id: str) -> bool:
    """Clear the pending clarification while preserving other memory fields."""
    return upsert_session_memory(session_id, {"pending_clarification": None})
