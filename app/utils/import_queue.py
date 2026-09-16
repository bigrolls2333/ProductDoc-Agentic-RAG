"""RQ enqueue adapter with a local-background-task fallback."""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.redis_utils import get_redis_client
from app.conf.rq_config import rq_config
from app.core.logger import logger


@dataclass
class EnqueueResult:
    """Result returned by an ingestion enqueue attempt."""

    enqueued: bool
    job_id: str
    reason: str = ""


def enqueue_import_task(
    task_id: str,
    local_dir: str,
    local_file_path: str,
    doc_id: str,
) -> EnqueueResult:
    """
    Enqueue the import graph into RQ when enabled.

    Returning enqueued=False is not an error. The API layer can fall back to
    FastAPI BackgroundTasks so local development still works without a worker.
    """
    if not rq_config.enabled:
        return EnqueueResult(enqueued=False, job_id="", reason="RQ_ENABLED is false")

    redis_client = get_redis_client()
    if redis_client is None:
        return EnqueueResult(enqueued=False, job_id="", reason="Redis is unavailable")

    try:
        from rq import Queue
    except ImportError:
        return EnqueueResult(enqueued=False, job_id="", reason="python package 'rq' is not installed")

    job_id = f"import:{task_id}"
    try:
        queue = Queue(rq_config.queue_name, connection=redis_client)
        job = queue.enqueue(
            "app.import_process.api.file_import_service.run_graph_task",
            task_id,
            local_dir,
            local_file_path,
            doc_id,
            job_id=job_id,
            job_timeout=rq_config.job_timeout_seconds,
            result_ttl=rq_config.result_ttl_seconds,
            failure_ttl=rq_config.failure_ttl_seconds,
        )
        logger.info(f"[{task_id}] Ingestion job enqueued: queue={rq_config.queue_name}, job_id={job.id}")
        return EnqueueResult(enqueued=True, job_id=job.id)
    except Exception as e:
        logger.warning(f"[{task_id}] RQ enqueue failed; caller should use the local fallback: {e}", exc_info=True)
        return EnqueueResult(enqueued=False, job_id="", reason=str(e))
