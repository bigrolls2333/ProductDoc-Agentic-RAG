"""HTTP API for conversational product-document retrieval."""

from pathlib import Path
import time
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.utils.query_trace_utils import append_query_trace
from app.clients.mongo_history_utils import *
from app.clients.mongo_session_memory_utils import clear_session_memory
from app.clients.run_state_utils import create_run, complete_run
from app.conf.checkpoint_config import is_checkpoint_enabled
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state

app = FastAPI(
    title="ProductDoc Agentic RAG Query Service",
    description="Context-aware product-document question answering service",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat.html", summary="Serve the chat client")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="Chat client is not available")
    return FileResponse(chat_html_path)

class QueryRequest(BaseModel):
    """Request payload for a product-document query."""
    query: str = Field(..., description="User query")
    session_id: str = Field(None, description="Conversation identifier")
    is_stream: bool = Field(False, description="Whether to stream the response")



@app.get("/health", summary="Service health check")
async def health():
    """Return a lightweight service health response."""
    return {"ok": True}


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    """Execute one query graph run and publish its final task state."""
    logger.info(f"Query graph started: session_id={session_id}, stream={is_stream}")

    set_task_result(session_id, "answer", "")
    checkpoint_enabled = is_checkpoint_enabled()
    run_id = create_run(
        session_id=session_id,
        graph_name="query_graph",
        checkpoint_backend="sqlite" if checkpoint_enabled else "none",
        resume_from_checkpoint=False,
    )

    default_state = create_query_default_state(
        original_query=user_query,
        session_id=session_id,
        is_stream=is_stream,
        trace_started_at=time.time(),
        thread_id=session_id,
        resume_from_checkpoint=False,
        checkpoint_enabled=checkpoint_enabled,
        run_id=run_id,
    )
    try:
        invoke_config = {"configurable": {"thread_id": session_id}}
        query_app.invoke(default_state, config=invoke_config)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        complete_run(run_id, status="completed")
    except Exception as e:
        logger.error(f"Query graph failed: {e}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        append_query_trace(default_state, status="failed", error=str(e))
        complete_run(run_id, status="failed", error=str(e))
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})



@app.post(
    "/query",
    summary="Run a product-document query",
    description="Submit a query synchronously or start a streamed response.",
)
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """Submit a query and return either the completed answer or a session handle."""
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    is_stream = request.is_stream
    if is_stream:
        create_sse_queue(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    logger.info(f"Query accepted: session_id={session_id}, stream={is_stream}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id,user_query,is_stream)
        return {
            "message":"Query accepted",
            "session_id":session_id
        }
    else:
        await run_in_threadpool(run_query_graph, session_id, user_query, is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"Query completed",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }



@app.get(
    "/stream/{session_id}",
    summary="Stream query events",
    description="Return server-sent events for an active query session.",
)
async def stream(session_id: str, request: Request):
    """Stream server-sent events for a query session."""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}", summary="Read conversation history")
async def history(session_id: str, limit: int = 50):
    """Return recent messages and retrieval metadata for a session."""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls": r.get("image_urls", []),
                "evidence_docs": r.get("evidence_docs", []),
                "route_meta": r.get("route_meta", {}),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}", summary="Clear conversation history")
async def clear_chat_history(session_id: str):
    """Delete persisted messages and session-level memory for a session."""
    count = clear_history(session_id)
    memory_deleted = clear_session_memory(session_id)
    return {
        "message": "History cleared",
        "deleted_count": count,
        "session_memory_deleted_count": memory_deleted,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
