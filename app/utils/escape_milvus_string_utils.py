"""Escape user-controlled values before placing them in Milvus filters."""

def escape_milvus_string(value: str) -> str:
    """Return a Milvus-filter-safe representation of ``value``."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
