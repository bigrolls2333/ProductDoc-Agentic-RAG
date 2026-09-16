"""Configuration for the local BGE-M3 embedding model."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # Local model path
    bge_m3: str       # Model identifier
    bge_device: str   # Runtime device
    bge_fp16: bool    # Whether to use half precision

embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
