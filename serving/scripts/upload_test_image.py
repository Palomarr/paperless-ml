"""Upload a test image to MinIO for HTR benchmarking."""

import io
import boto3
from PIL import Image

MINIO_ENDPOINT = "http://localhost:9000"
BUCKET = "paperless-images"
KEY = "documents/a3f7c2e1/regions/test.png"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

# Create a 224x224 white test image (same as old benchmark used)
img = Image.new("RGB", (224, 224), "white")
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

s3.put_object(Bucket=BUCKET, Key=KEY, Body=buf.getvalue(), ContentType="image/png")
print(f"Uploaded test image to s3://{BUCKET}/{KEY}")
