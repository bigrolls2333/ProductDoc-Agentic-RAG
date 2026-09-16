"""HTTP API for asynchronous product-document ingestion."""

import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    set_task_result,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.utils.file_hash_utils import build_doc_id
from app.utils.import_queue import enqueue_import_task
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import get_kb_import_app  # Lazy workflow compilation
from app.utils.redis_lock_utils import acquire_upload_lock, release_upload_lock
from app.lm.embedding_utils import get_bge_m3_ef
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

app = FastAPI(
    title="ProductDoc Document Import Service",
    description="Asynchronous ingestion pipeline for product documents and vector indexing.",
)


@app.on_event("startup")
async def warmup_local_models():
    """Initialize local models and required Milvus collections at startup."""
    try:
        logger.info("Startup warmup: initializing BGE-M3")
        get_bge_m3_ef()
        logger.info("Startup warmup: BGE-M3 is ready")

        logger.info("Startup warmup: checking Milvus collections")
        client = get_milvus_client()
        if not client:
            raise ValueError("Milvus client is unavailable")
        from app.import_process.agent.nodes.node_import_milvus import create_collection
        from app.import_process.agent.nodes.node_item_name_recognition import _ensure_item_name_collection_ready

        chunks_collection = milvus_config.chunks_collection
        item_name_collection = milvus_config.item_name_collection
        vector_dim = 1024

        if chunks_collection:
            if client.has_collection(collection_name=chunks_collection):
                logger.info(f"Startup warmup: reusing collection {chunks_collection}")
            else:
                create_collection(client, chunks_collection, vector_dim)

        if item_name_collection:
            _ensure_item_name_collection_ready(client, item_name_collection)

        logger.info("Startup warmup: Milvus collections are ready")
    except Exception as e:
        logger.error(f"Startup warmup failed: {e}", exc_info=True)
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Return a lightweight service health response."""
    return {"ok": True}


@app.get("/import.html", response_class=FileResponse, summary="Serve the import client")
async def get_import_page():
    """Serve the bundled document-import client."""
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    logger.debug(f"Serving import client from {html_abs_path}")

    if not os.path.exists(html_abs_path):
        logger.error(f"Import client not found: {html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    return FileResponse(
        path=html_abs_path,
        media_type="text/html",
    )


def run_graph_task(task_id: str, local_dir: str, local_file_path: str, doc_id: str = ""):
    """Run one ingestion task and publish node-level progress."""
    try:
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] Ingestion started: path={local_file_path}")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["doc_id"] = doc_id

        kb_import_app = get_kb_import_app()
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] Ingestion node completed: {node_name}")
                add_done_task(task_id, node_name)

        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] Ingestion completed")

    except Exception as e:
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] Ingestion failed: {e}", exc_info=True)
    finally:
        if doc_id:
            release_upload_lock(doc_id=doc_id, task_id=task_id)


@app.post(
    "/upload",
    summary="Upload product documents",
    description="Accept PDF or Markdown files and enqueue asynchronous ingestion jobs.",
)
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Persist uploads, enforce document-level ownership, and enqueue ingestion."""
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_ids = []
    response_items = []

    for file in files:
        task_id = str(uuid.uuid4())
        logger.info(f"[{task_id}] Upload received: filename={file.filename}, content_type={file.content_type}")

        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] Upload persisted locally: {local_file_abs_path}")

        doc_id = build_doc_id(local_file_abs_path)
        lock_result = acquire_upload_lock(doc_id=doc_id, task_id=task_id)
        if not lock_result.acquired:
            existing_task_id = lock_result.owner_task_id
            logger.info(
                f"[{task_id}] Duplicate document detected: doc_id={doc_id}, owner={existing_task_id}"
            )
            response_items.append({
                "filename": file.filename,
                "doc_id": doc_id,
                "task_id": existing_task_id,
                "status": "duplicate_in_progress",
            })
            if existing_task_id:
                task_ids.append(existing_task_id)
            continue

        task_ids.append(task_id)
        update_task_status(task_id, "pending")
        set_task_result(task_id, "doc_id", doc_id)

        add_running_task(task_id, "upload_file")

        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            minio_client = get_minio_client()
            if minio_client is None:
                raise HTTPException(status_code=500,
                                    detail="MinIO service connection failed, please check MinIO config")
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")

            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type,
            )
            logger.info(f"[{task_id}] Object storage upload completed: {minio_object_name}")
        except Exception as e:
            logger.warning(f"[{task_id}] Object storage upload failed; continuing locally: {e}", exc_info=True)

        add_done_task(task_id, "upload_file")

        enqueue_result = enqueue_import_task(task_id, task_local_dir, local_file_abs_path, doc_id)
        if enqueue_result.enqueued:
            backend = "rq"
            job_id = enqueue_result.job_id
        else:
            backend = "background_tasks"
            job_id = ""
            background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path, doc_id)
            logger.info(f"[{task_id}] Falling back to in-process background task: {enqueue_result.reason}")

        response_items.append({
            "filename": file.filename,
            "doc_id": doc_id,
            "task_id": task_id,
            "status": "accepted",
            "backend": backend,
            "job_id": job_id,
        })

    logger.info(f"Upload batch accepted: count={len(files)}, task_ids={task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids,
        "items": response_items,
    }


@app.get(
    "/status/{task_id}",
    summary="Read ingestion status",
    description="Return task status and completed or active pipeline stages.",
)
async def get_task_progress(task_id: str):
    """Return the current status and stage progress for an ingestion task."""
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }
    status = task_status_info["status"]
    done_list = task_status_info["done_list"]
    running_list = task_status_info["running_list"]
    if status in {"processing", "completed", "failed"} or done_list or running_list:
        logger.debug(
            f"[{task_id}] Status queried: status={status}, done={done_list}, running={running_list}"
        )
    return task_status_info


if __name__ == "__main__":
    """Run the import service for local development."""
    logger.info("Document import service starting")
    uvicorn.run(
        app=app,
        host="127.0.0.1",
        port=8000,
    )
