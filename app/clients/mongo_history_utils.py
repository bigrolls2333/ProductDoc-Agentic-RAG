"""MongoDB persistence helpers for conversation history."""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class HistoryMongoTool:
    """Manage the MongoDB collection used for conversation history."""
    def __init__(self):
        """Create the client, select the collection, and ensure its index."""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Conversation history connected: {self.db_name}.chat_message")
        except Exception as e:
            logging.error(f"Conversation history initialization failed: {e}")
            raise


_history_mongo_tool = None
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")

def get_history_mongo_tool() -> HistoryMongoTool:
    """Return the cached history tool, creating it on first access."""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool



def clear_history(session_id: str) -> int:
    """Delete all history records for a session and return the delete count."""
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        evidence_docs: List[dict] = None,
        route_meta: dict = None,
        message_id: str = None
) -> str:
    """Insert or update one conversation message and return its identifier."""
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names or [],
        "image_urls": image_urls or [],
        "evidence_docs": evidence_docs or [],
        "route_meta": route_meta or {},
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """Update product names for a batch of history records."""
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids}
            },
            {"$set": {"item_names": item_names}},
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent session messages in chronological order."""
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}

        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        messages = list(cursor)
        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []
