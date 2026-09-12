from __future__ import annotations

from dataclasses import dataclass

from app.clients.redis_utils import get_redis_client
from app.conf.redis_config import redis_config
from app.core.logger import logger


@dataclass
class UploadLockResult:
    acquired: bool
    lock_enabled: bool
    owner_task_id: str


class RedisUploadLock:
    def __init__(self) -> None:
        self.prefix = redis_config.key_prefix.strip(":") or "rag"
        self.ttl_seconds = redis_config.upload_lock_ttl_seconds

    def _key(self, doc_id: str) -> str:
        return f"{self.prefix}:lock:upload:{doc_id}"

    def acquire(self, doc_id: str, task_id: str) -> UploadLockResult:
        client = get_redis_client()
        if client is None:
            return UploadLockResult(acquired=True, lock_enabled=False, owner_task_id=task_id)

        key = self._key(doc_id)
        try:
            acquired = bool(client.set(key, task_id, nx=True, ex=self.ttl_seconds))
            if acquired:
                return UploadLockResult(acquired=True, lock_enabled=True, owner_task_id=task_id)

            owner_task_id = client.get(key) or ""
            return UploadLockResult(acquired=False, lock_enabled=True, owner_task_id=owner_task_id)
        except Exception as e:
            logger.warning(f"Redis上传锁获取失败，继续按无锁模式处理：doc_id={doc_id} error={e}")
            return UploadLockResult(acquired=True, lock_enabled=False, owner_task_id=task_id)

    def release(self, doc_id: str, task_id: str) -> None:
        client = get_redis_client()
        if client is None:
            return

        key = self._key(doc_id)
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        end
        return 0
        """
        try:
            client.eval(script, 1, key, task_id)
        except Exception as e:
            logger.warning(f"Redis上传锁释放失败：doc_id={doc_id} task_id={task_id} error={e}")


_upload_lock = RedisUploadLock()


def acquire_upload_lock(doc_id: str, task_id: str) -> UploadLockResult:
    return _upload_lock.acquire(doc_id=doc_id, task_id=task_id)


def release_upload_lock(doc_id: str, task_id: str) -> None:
    _upload_lock.release(doc_id=doc_id, task_id=task_id)
