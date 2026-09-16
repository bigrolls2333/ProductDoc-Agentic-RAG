"""Run metadata used for query observability and checkpoint correlation."""

import time
import uuid
from typing import Dict, Any, Optional

from app.core.logger import logger

# In-memory state is always available for the active process.
_run_states: Dict[str, Dict[str, Any]] = {}

# MongoDB persistence is optional.
_mongo_available: Optional[bool] = None


def _try_get_mongo_collection():
    """Return the optional MongoDB collection when it is available."""
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
    """Create a run record and return its identifier."""
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

    _run_states[run_id] = doc

    collection = _try_get_mongo_collection()
    if collection is not None:
        try:
            collection.insert_one(doc)
        except Exception as e:
            logger.debug(f"Run metadata persistence failed: {e}")

    logger.info(f"Run created: run_id={run_id}, session={session_id}, graph={graph_name}")
    return run_id


def update_run_node(run_id: str, node_name: str) -> None:
    """Record the latest graph node for a run."""
    if run_id in _run_states:
        _run_states[run_id]["current_node"] = node_name
        _run_states[run_id]["updated_at"] = time.time()

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
    """Record the most recent checkpoint timestamp for a run."""
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
    """Mark a run as completed or failed and persist its terminal metadata."""
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

    logger.info(f"Run finished: run_id={run_id}, status={status}" + (f", error={error}" if error else ""))


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a run record from the local metadata store."""
    return _run_states.get(run_id)


def get_session_runs(session_id: str, limit: int = 10) -> list:
    """Return the most recent run records for a session."""
    results = [
        r for r in _run_states.values()
        if r.get("session_id") == session_id
    ]
    results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return results[:limit]
