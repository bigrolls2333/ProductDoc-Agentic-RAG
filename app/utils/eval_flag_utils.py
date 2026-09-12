import os


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def is_eval_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in TRUE_VALUES


def eval_mode(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip().lower()
