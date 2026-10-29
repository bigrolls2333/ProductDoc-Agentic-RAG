"""Provider-specific options for explicitly requesting non-thinking responses."""

from urllib.parse import urlparse


def non_thinking_body(base_url):
    host = (urlparse(base_url or "").hostname or "").lower()
    if host == "api.deepseek.com":
        return {"thinking": {"type": "disabled"}}
    if host == "api.openai.com":
        return {}
    return {"enable_thinking": False}
