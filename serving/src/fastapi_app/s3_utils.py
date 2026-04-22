"""MinIO / S3 helpers used by the FastAPI serving app.

Exposes:
  - download_image_from_s3(url) — fetch a single image (PIL.Image)
  - download_file_from_s3(url, local_path) — fetch a single object to disk
  - download_prefix_from_s3(url, local_dir) — fetch every object under
    a prefix into local_dir, preserving relative paths. Used by app_ort.py
    at boot to pull fine-tuned ONNX checkpoints from the warehouse bucket
    when HTR_MODEL_URI / EMBEDDING_MODEL_URI env vars are set.

MinIO client credentials default to admin/paperless_minio (predates the
minioadmin rename) but are always overridden via MINIO_ACCESS_KEY /
MINIO_SECRET_KEY env vars in compose. Don't rely on the defaults.
"""
from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from PIL import Image

log = logging.getLogger(__name__)

_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "paperless_minio"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
)


def _parse_s3_uri(s3_url: str) -> tuple[str, str]:
    """Split s3://bucket/key/... into (bucket, key). Trailing slash preserved."""
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"expected s3:// URI, got {s3_url!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"missing bucket in {s3_url!r}")
    return bucket, key


def download_image_from_s3(s3_url: str) -> Image.Image:
    bucket, key = _parse_s3_uri(s3_url)
    response = _client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return Image.open(BytesIO(data))


def download_file_from_s3(s3_url: str, local_path: str | os.PathLike) -> None:
    """Download a single object to a local path, creating parent dirs."""
    bucket, key = _parse_s3_uri(s3_url)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading s3://%s/%s → %s", bucket, key, local_path)
    _client.fget_object(bucket, key, str(local_path))


def download_prefix_from_s3(s3_url: str, local_dir: str | os.PathLike) -> int:
    """Download every object under an s3://bucket/prefix/ into local_dir,
    preserving relative paths. Idempotent: re-downloads overwrite.

    Returns: number of objects downloaded.
    """
    bucket, prefix = _parse_s3_uri(s3_url)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for obj in _client.list_objects(bucket, prefix=prefix, recursive=True):
        rel = obj.object_name[len(prefix):]
        if not rel:  # skip the directory marker itself
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "downloading s3://%s/%s → %s (%d bytes)",
            bucket, obj.object_name, dest, obj.size,
        )
        _client.fget_object(bucket, obj.object_name, str(dest))
        count += 1

    if count == 0:
        raise RuntimeError(
            f"no objects found under s3://{bucket}/{prefix} — check "
            f"the URI and that the training/export step uploaded artifacts"
        )
    log.info("downloaded %d object(s) from s3://%s/%s into %s",
             count, bucket, prefix, local_dir)
    return count
