"""Sliding-window rate limiting for outbound API calls."""

import time
from typing import Deque
from app.core.logger import logger


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    """Block until the sliding-window request budget has capacity."""
    current_time = time.time()

    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    if len(request_times) >= max_requests:
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(
                f"Rate limit reached: max_requests={max_requests}, "
                f"window_seconds={window_seconds}, wait_seconds={sleep_duration:.2f}"
            )
            time.sleep(sleep_duration)
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    request_times.append(current_time)
    logger.debug(f"Rate-limit window updated: request_count={len(request_times)}")
