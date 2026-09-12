from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass
class RQConfig:
    enabled: bool
    queue_name: str
    job_timeout_seconds: int
    result_ttl_seconds: int
    failure_ttl_seconds: int


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


rq_config = RQConfig(
    enabled=_get_bool("RQ_ENABLED", False),
    queue_name=os.getenv("RQ_IMPORT_QUEUE_NAME", "rag-import"),
    job_timeout_seconds=_get_int("RQ_JOB_TIMEOUT_SECONDS", 60 * 60),
    result_ttl_seconds=_get_int("RQ_RESULT_TTL_SECONDS", 24 * 60 * 60),
    failure_ttl_seconds=_get_int("RQ_FAILURE_TTL_SECONDS", 7 * 24 * 60 * 60),
)
