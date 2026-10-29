"""Run one evaluation with an external API env file, without copying secrets."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--no-web", action="store_true", help="disable Web in graph nodes as well as planner")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from dotenv import dotenv_values, load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    values = dotenv_values(args.env_file)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_DEFAULT_TEMPERATURE"):
        if values.get(key):
            os.environ[key] = values[key]
    base_url = values.get("OPENAI_BASE_URL") or values.get("OPENAI_API_BASE")
    model = values.get("LLM_DEFAULT_MODEL") or values.get("MODEL_NAME")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if model:
        os.environ["LLM_DEFAULT_MODEL"] = model
    if args.no_web:
        os.environ["EVAL_DISABLE_WEB"] = "1"
    if args.no_checkpoint:
        os.environ["CHECKPOINT_ENABLED"] = "false"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("EVAL_DISABLE_PRODUCT_MEMORY_WRITE", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    script = Path(args.script).resolve()
    if root not in script.parents or not script.is_file():
        raise ValueError("script must be an existing file in this project")
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(root))
    sys.argv = [str(script)] + args.args
    print(f"Evaluation API model: {os.environ.get('LLM_DEFAULT_MODEL', 'project default')}", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
