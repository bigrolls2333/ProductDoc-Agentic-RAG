from __future__ import annotations

from typing import Optional

from app.conf.redis_config import redis_config
from app.core.logger import logger


_redis_client = None
_redis_unavailable_reason: Optional[str] = None


def is_redis_enabled() -> bool:
    return redis_config.enabled


def get_redis_client():
    """
    Return a singleton Redis client when Redis is enabled.

    Redis is optional in this project: if the dependency is missing, the service
    is disabled, or the server cannot be reached, callers should fall back to
    the in-memory task store.
    """
    global _redis_client, _redis_unavailable_reason

    if not redis_config.enabled:
        _redis_unavailable_reason = "REDIS_ENABLED is false"
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
    except ImportError:
        _redis_unavailable_reason = "python package 'redis' is not installed"
        logger.warning(f"Redis未启用：{_redis_unavailable_reason}")
        return None

    try:
        if redis_config.url:
            client = redis.Redis.from_url(
                redis_config.url,
                decode_responses=True,
                socket_timeout=redis_config.socket_timeout_seconds,
                socket_connect_timeout=redis_config.socket_timeout_seconds,
            )
        else:
            client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password,
                decode_responses=True,
                socket_timeout=redis_config.socket_timeout_seconds,
                socket_connect_timeout=redis_config.socket_timeout_seconds,
            )

        client.ping()
        _redis_client = client
        _redis_unavailable_reason = None
        logger.info("Redis客户端连接成功，任务状态将写入Redis")
        return _redis_client
    except Exception as e:
        _redis_unavailable_reason = str(e)
        logger.warning(f"Redis连接不可用，任务状态回退到内存：{e}")
        return None


def get_redis_unavailable_reason() -> str:
    return _redis_unavailable_reason or ""


def check_redis_health() -> bool:
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception as e:
        logger.warning(f"Redis健康检查失败：{e}")
        return False
