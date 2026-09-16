"""Configuration for the Milvus vector store."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus service endpoint
    chunks_collection: str   # Document chunk collection
    item_name_collection: str  # Product-name collection

milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)
