from __future__ import annotations

from typing import Dict, List, Protocol

from app.clients.redis_utils import get_redis_client
from app.conf.redis_config import redis_config
from app.core.logger import logger


class TaskStore(Protocol):
    def add_running(self, task_id: str, node_name: str) -> None: ...
    def add_done(self, task_id: str, node_name: str) -> None: ...
    def set_result(self, task_id: str, key: str, value: str) -> None: ...
    def get_result(self, task_id: str, key: str, default: str = "") -> str: ...
    def get_status(self, task_id: str) -> str: ...
    def get_done(self, task_id: str) -> List[str]: ...
    def get_running(self, task_id: str) -> List[str]: ...
    def update_status(self, task_id: str, status_name: str) -> None: ...
    def clear(self, task_id: str) -> None: ...


class InMemoryTaskStore:
    def __init__(self) -> None:
        self._running: Dict[str, List[str]] = {}
        self._done: Dict[str, List[str]] = {}
        self._status: Dict[str, str] = {}
        self._result: Dict[str, Dict[str, str]] = {}

    def _ensure_task(self, task_id: str) -> None:
        self._running.setdefault(task_id, [])
        self._done.setdefault(task_id, [])
        self._result.setdefault(task_id, {})

    def add_running(self, task_id: str, node_name: str) -> None:
        self._ensure_task(task_id)
        running = self._running[task_id]
        if node_name not in running:
            running.append(node_name)

    def add_done(self, task_id: str, node_name: str) -> None:
        self._ensure_task(task_id)
        self._running[task_id] = [n for n in self._running[task_id] if n != node_name]
        done = self._done[task_id]
        if node_name not in done:
            done.append(node_name)

    def set_result(self, task_id: str, key: str, value: str) -> None:
        self._ensure_task(task_id)
        self._result[task_id][key] = value

    def get_result(self, task_id: str, key: str, default: str = "") -> str:
        self._ensure_task(task_id)
        return self._result.get(task_id, {}).get(key, default)

    def get_status(self, task_id: str) -> str:
        return self._status.get(task_id, "")

    def get_done(self, task_id: str) -> List[str]:
        self._ensure_task(task_id)
        return list(self._done.get(task_id, []))

    def get_running(self, task_id: str) -> List[str]:
        self._ensure_task(task_id)
        return list(self._running.get(task_id, []))

    def update_status(self, task_id: str, status_name: str) -> None:
        self._status[task_id] = status_name

    def clear(self, task_id: str) -> None:
        self._running.pop(task_id, None)
        self._done.pop(task_id, None)
        self._status.pop(task_id, None)
        self._result.pop(task_id, None)


class RedisTaskStore:
    def __init__(self, client) -> None:
        self.client = client
        self.prefix = redis_config.key_prefix.strip(":") or "rag"
        self.ttl_seconds = redis_config.task_ttl_seconds

    def _key(self, task_id: str, suffix: str) -> str:
        return f"{self.prefix}:task:{task_id}:{suffix}"

    def _meta_key(self, task_id: str) -> str:
        return self._key(task_id, "meta")

    def _running_key(self, task_id: str) -> str:
        return self._key(task_id, "running")

    def _done_set_key(self, task_id: str) -> str:
        return self._key(task_id, "done:set")

    def _done_order_key(self, task_id: str) -> str:
        return self._key(task_id, "done:order")

    def _result_key(self, task_id: str) -> str:
        return self._key(task_id, "result")

    def _touch(self, pipe, task_id: str) -> None:
        for key in [
            self._meta_key(task_id),
            self._running_key(task_id),
            self._done_set_key(task_id),
            self._done_order_key(task_id),
            self._result_key(task_id),
        ]:
            pipe.expire(key, self.ttl_seconds)

    def add_running(self, task_id: str, node_name: str) -> None:
        pipe = self.client.pipeline()
        pipe.sadd(self._running_key(task_id), node_name)
        pipe.hset(self._meta_key(task_id), mapping={"last_node": node_name})
        self._touch(pipe, task_id)
        pipe.execute()

    def add_done(self, task_id: str, node_name: str) -> None:
        done_set_key = self._done_set_key(task_id)
        pipe = self.client.pipeline()
        pipe.srem(self._running_key(task_id), node_name)
        pipe.sadd(done_set_key, node_name)
        pipe.execute()

        if self.client.sismember(done_set_key, node_name):
            order_key = self._done_order_key(task_id)
            if node_name not in self.client.lrange(order_key, 0, -1):
                self.client.rpush(order_key, node_name)

        pipe = self.client.pipeline()
        pipe.hset(self._meta_key(task_id), mapping={"last_node": node_name})
        self._touch(pipe, task_id)
        pipe.execute()

    def set_result(self, task_id: str, key: str, value: str) -> None:
        pipe = self.client.pipeline()
        pipe.hset(self._result_key(task_id), key, value)
        self._touch(pipe, task_id)
        pipe.execute()

    def get_result(self, task_id: str, key: str, default: str = "") -> str:
        value = self.client.hget(self._result_key(task_id), key)
        return value if value is not None else default

    def get_status(self, task_id: str) -> str:
        return self.client.hget(self._meta_key(task_id), "status") or ""

    def get_done(self, task_id: str) -> List[str]:
        return self.client.lrange(self._done_order_key(task_id), 0, -1)

    def get_running(self, task_id: str) -> List[str]:
        return sorted(self.client.smembers(self._running_key(task_id)))

    def update_status(self, task_id: str, status_name: str) -> None:
        pipe = self.client.pipeline()
        pipe.hset(self._meta_key(task_id), mapping={"status": status_name})
        self._touch(pipe, task_id)
        pipe.execute()

    def clear(self, task_id: str) -> None:
        self.client.delete(
            self._meta_key(task_id),
            self._running_key(task_id),
            self._done_set_key(task_id),
            self._done_order_key(task_id),
            self._result_key(task_id),
        )


class ResilientTaskStore:
    def __init__(self, primary: TaskStore | None, fallback: InMemoryTaskStore) -> None:
        self.primary = primary
        self.fallback = fallback
        self._primary_failed = False

    def _run(self, method: str, *args, **kwargs):
        if self.primary is not None and not self._primary_failed:
            try:
                return getattr(self.primary, method)(*args, **kwargs)
            except Exception as e:
                self._primary_failed = True
                logger.warning(f"Redis任务状态操作失败，后续回退到内存：{e}")
        return getattr(self.fallback, method)(*args, **kwargs)

    def add_running(self, task_id: str, node_name: str) -> None:
        self._run("add_running", task_id, node_name)

    def add_done(self, task_id: str, node_name: str) -> None:
        self._run("add_done", task_id, node_name)

    def set_result(self, task_id: str, key: str, value: str) -> None:
        self._run("set_result", task_id, key, value)

    def get_result(self, task_id: str, key: str, default: str = "") -> str:
        return self._run("get_result", task_id, key, default)

    def get_status(self, task_id: str) -> str:
        return self._run("get_status", task_id)

    def get_done(self, task_id: str) -> List[str]:
        return self._run("get_done", task_id)

    def get_running(self, task_id: str) -> List[str]:
        return self._run("get_running", task_id)

    def update_status(self, task_id: str, status_name: str) -> None:
        self._run("update_status", task_id, status_name)

    def clear(self, task_id: str) -> None:
        self._run("clear", task_id)


_memory_store = InMemoryTaskStore()
_task_store: ResilientTaskStore | None = None


def get_task_store() -> ResilientTaskStore:
    global _task_store
    if _task_store is None:
        redis_client = get_redis_client()
        primary = RedisTaskStore(redis_client) if redis_client is not None else None
        if primary is None:
            logger.info("任务状态使用内存存储；如需Redis，请配置 REDIS_ENABLED=True 并启动Redis")
        _task_store = ResilientTaskStore(primary=primary, fallback=_memory_store)
    return _task_store
