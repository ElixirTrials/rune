"""Optional S3 uploader for bin manifests.

The uploader is a pure add-on: local paths remain the source of truth. When
``boto3`` is unavailable or AWS credentials are missing, :func:`upload_manifest`
logs a warning and returns ``False`` without raising.

``boto3`` is imported inside :func:`upload_manifest` so this module stays
importable even if boto3 is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["build_s3_key", "upload_manifest"]


def build_s3_key(prefix: str, basename: str) -> str:
    """Join an S3 prefix and basename into a single key.

    Normalizes a trailing slash on the prefix and returns the basename alone
    when the prefix is empty.

    Args:
        prefix: S3 key prefix (e.g. ``"oracles/run-1"``). May be empty.
        basename: Object basename (e.g. ``"decompose_humaneval.jsonl"``).

    Returns:
        A single ``"<prefix>/<basename>"`` key, or just ``basename`` when
        prefix is empty.
    """
    if not prefix:
        return basename
    return f"{prefix.rstrip('/')}/{basename}"


def upload_manifest(manifest: Path, bucket: str, prefix: str) -> bool:
    """Upload a single manifest file to S3.

    Returns ``False`` and logs a warning when boto3 is missing or AWS
    credentials are unavailable — upload is an optional add-on and must never
    break the corpus producer run.

    Args:
        manifest: Local path to the JSONL manifest to upload.
        bucket: Target S3 bucket name.
        prefix: Key prefix within the bucket (no leading ``s3://``).

    Returns:
        True on a successful upload, False on any import/credentials/client
        error.
    """
    try:
        import boto3  # noqa: PLC0415
    except ImportError:
        logger.warning("boto3 not available; skipping S3 upload of %s", manifest)
        return False

    if boto3 is None:
        logger.warning("boto3 not available; skipping S3 upload of %s", manifest)
        return False

    try:
        from botocore.exceptions import (  # noqa: PLC0415
            BotoCoreError,
            ClientError,
            NoCredentialsError,
        )
    except ImportError:
        logger.warning("botocore not available; skipping S3 upload of %s", manifest)
        return False

    key = build_s3_key(prefix, manifest.name)
    try:
        client = boto3.client("s3")
        client.upload_file(str(manifest), bucket, key)
    except NoCredentialsError:
        logger.warning("AWS credentials not found; skipping S3 upload of %s", manifest)
        return False
    except (ClientError, BotoCoreError) as exc:
        logger.warning(
            "S3 upload of %s to s3://%s/%s failed: %s",
            manifest,
            bucket,
            key,
            exc,
        )
        return False

    logger.info("Uploaded %s to s3://%s/%s", manifest, bucket, key)
    return True
