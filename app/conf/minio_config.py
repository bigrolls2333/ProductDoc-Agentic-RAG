"""Configuration for the MinIO object store."""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MinIOConfig:
    endpoint: str    # Service endpoint
    access_key: str  # Access key
    secret_key: str  # Secret key
    bucket_name: str # Default bucket
    minio_img_dir: str # Image prefix
    minio_secure: bool # Whether TLS is enabled

minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
