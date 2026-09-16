"""Project path resolution helpers."""

from pathlib import Path
from dotenv import load_dotenv
import os

def get_path_dir(ps:int = 0)->Path:
    """Return the directory ``ps`` levels above this module."""
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    """Resolve the project root from ``PROJECT_ROOT`` or an ancestor marker."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"Project root marker not found: {identifier}")


PROJECT_ROOT = get_project_root(".env")
