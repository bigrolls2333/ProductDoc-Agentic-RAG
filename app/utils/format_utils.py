"""JSON serialization helpers used by workflow and API tooling."""

import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:
    """Serialize workflow state as UTF-8 JSON for logs or diagnostics."""
    return json.dumps(state, indent=indent, ensure_ascii=False)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """Serialize any JSON-compatible value with the requested formatting."""
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

