"""Configure optional SQLite persistence for LangGraph checkpoints."""

import os
import sqlite3
from pathlib import Path
from typing import Optional

from app.core.logger import logger

CHECKPOINT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "checkpoints.db"

_checkpointer_instance: Optional[object] = None


def get_checkpointer() -> Optional[object]:
    """Return the cached SQLite checkpointer, initializing it on first access."""
    global _checkpointer_instance

    enabled = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.info("Checkpointing disabled by CHECKPOINT_ENABLED=false")
        return None

    if _checkpointer_instance is not None:
        return _checkpointer_instance

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
        _checkpointer_instance = SqliteSaver(conn)
        logger.info(f"SQLite checkpointer initialized: {CHECKPOINT_DB_PATH}")
        return _checkpointer_instance

    except ImportError:
        logger.warning(
            "SQLite checkpointer dependency is unavailable; continuing without checkpoint persistence"
        )
        return None
    except Exception as e:
        logger.warning(f"SQLite checkpointer initialization failed; continuing without persistence: {e}")
        return None


def close_checkpointer() -> None:
    """Close the checkpointer database connection during service shutdown."""
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        try:
            conn = getattr(_checkpointer_instance, "conn", None)
            if conn:
                conn.close()
            _checkpointer_instance = None
            logger.info("Checkpointer closed")
        except Exception as e:
            logger.warning(f"Failed to close checkpointer: {e}")


def is_checkpoint_enabled() -> bool:
    """Return whether checkpoint persistence is enabled and initialized."""
    return get_checkpointer() is not None
