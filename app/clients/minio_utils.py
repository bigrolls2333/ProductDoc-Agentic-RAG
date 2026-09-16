"""MinIO client lifecycle and bucket policy helpers."""

import os
import json

from minio import Minio

from app.conf.minio_config import minio_config
from app.core.logger import logger

# Process-local client cache.
_minio_client = None


def _create_minio_client() -> Minio:
    """Create a MinIO client from the application configuration."""
    return Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
    )


def _set_bucket_policy(bucket_name: str) -> str:
    """Build the public-read S3 policy used for rendered document images."""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    return json.dumps(policy)


def _create_bucket_ready(client: Minio):
    """Ensure the configured bucket exists and has the image access policy."""
    bucket_name = minio_config.bucket_name
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
        logger.info(f"Created MinIO bucket: {bucket_name}")
    else:
        logger.debug(f"Reusing MinIO bucket: {bucket_name}")


def get_minio_client() -> Minio:
    """Return the cached MinIO client, initializing its bucket on first use."""
    global _minio_client

    if _minio_client is None:
        logger.info("Initializing MinIO client")
        client = _create_minio_client()
        _create_bucket_ready(client)
        _minio_client = client
        logger.info("MinIO client initialized")

    return _minio_client
