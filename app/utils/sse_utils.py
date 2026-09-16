"""Server-sent event queue management for streamed query responses."""

import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request
from app.core.logger import logger


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"
    CLOSE = "__close__"


# Process-local queues keyed by session identifier.
_session_stream: Dict[str, queue.Queue] = {}

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """Return the queue registered for a session, if present."""
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """Create and register a queue for a streamed session."""
    logger.debug(f"SSE queue created: session_id={session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id: str):
    """Remove the queue registered for a session."""
    logger.debug(f"SSE queue removed: session_id={session_id}")
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """Serialize an event using the SSE wire format."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """Publish an event to a session queue when the stream is active."""
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})
    else:
        logger.debug(f"SSE event dropped because no queue is registered: session_id={session_id}, event={event}")

async def sse_generator(session_id: str, request: Request):
    """Yield server-sent events until the client or producer closes the stream."""
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        logger.warning(f"SSE queue not found: session_id={session_id}")
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack("ready", {})

        while True:
            if await request.is_disconnected():
                logger.debug(f"SSE client disconnected: session_id={session_id}")
                break

            try:
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            if event == "__close__":
                logger.debug(f"SSE close event received: session_id={session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        logger.debug(f"SSE stream cancelled: session_id={session_id}")
        return
    except Exception as e:
        logger.error(f"SSE stream failed: session_id={session_id}, error={e}", exc_info=True)
    finally:
        logger.debug(f"SSE stream finished: session_id={session_id}")
        remove_sse_queue(session_id)
