"""Configuration for Redis-backed task state and coordination."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass
class RedisConfig:
    enabled: bool
    url: str
    host: str
    port: int
    db: int
    password: str | None
    key_prefix: str
    task_ttl_seconds: int
    upload_lock_ttl_seconds: int
    socket_timeout_seconds: float


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


redis_config = RedisConfig(
    enabled=_get_bool("REDIS_ENABLED", False),
    url=os.getenv("REDIS_URL", ""),
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=_get_int("REDIS_PORT", 6379),
    db=_get_int("REDIS_DB", 0),
    password=os.getenv("REDIS_PASSWORD") or None,
    key_prefix=os.getenv("REDIS_KEY_PREFIX", "rag"),
    task_ttl_seconds=_get_int("REDIS_TASK_TTL_SECONDS", 24 * 60 * 60),
    upload_lock_ttl_seconds=_get_int("REDIS_UPLOAD_LOCK_TTL_SECONDS", 60 * 60),
    socket_timeout_seconds=_get_float("REDIS_SOCKET_TIMEOUT_SECONDS", 1.0),
)
