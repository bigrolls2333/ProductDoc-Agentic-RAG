from app.clients.redis_utils import get_redis_client
from app.conf.rq_config import rq_config
from app.core.logger import logger


def main() -> None:
    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis is required to run the RQ import worker")

    try:
        from rq import Queue, SimpleWorker
    except ImportError as exc:
        raise RuntimeError("python package 'rq' is required to run the worker") from exc

    queue = Queue(rq_config.queue_name, connection=redis_client)
    worker = SimpleWorker([queue], connection=redis_client)
    logger.info(f"RQ导入Worker启动：queue={rq_config.queue_name}")
    worker.work()


if __name__ == "__main__":
    main()
