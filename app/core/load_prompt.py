"""Load and render prompt templates from the repository prompt directory."""

from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger

def load_prompt(name: str, **kwargs) -> str:
    """Load a prompt template and render its format placeholders."""
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path.absolute()}")

    raw_prompt = prompt_path.read_text(encoding='utf-8')

    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"Prompt rendered: variables={list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt
