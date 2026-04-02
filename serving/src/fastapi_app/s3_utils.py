import os
from io import BytesIO
from urllib.parse import urlparse

from minio import Minio
from PIL import Image

_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "paperless_minio"),
    secure=False,
)


def download_image_from_s3(s3_url: str) -> Image.Image:
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    response = _client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    return Image.open(BytesIO(data))
