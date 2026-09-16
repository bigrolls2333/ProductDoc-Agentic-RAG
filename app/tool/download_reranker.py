"""Download the BGE reranker model to the configured local cache."""

import os

from modelscope.hub.snapshot_download import snapshot_download


cache_dir = os.getenv("BGE_RERANKER_CACHE_DIR", "models")
model_dir = snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=cache_dir,
)
print(f"BGE reranker model downloaded to: {model_dir}")
