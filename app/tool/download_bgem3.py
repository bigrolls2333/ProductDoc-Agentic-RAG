"""Download the BGE-M3 model to the configured local cache."""

import os

from modelscope.hub.snapshot_download import snapshot_download

cache_dir = os.getenv("BGE_M3_CACHE_DIR", "models")
model_dir = snapshot_download("BAAI/bge-m3", cache_dir=cache_dir)
print(f"BGE-M3 model downloaded to: {model_dir}")
