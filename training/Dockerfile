# HTR fine-tune image — CUDA 11.8 so P100 (compute 6.0) is supported.
# Matches ml-gateway's Dockerfile.fastapi-ort base so ONNX artifacts
# produced here load in the serving container without version drift.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# P100 compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.2.2 && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download base TrOCR so the first training step doesn't stall on
# HuggingFace Hub. Small image cost (~400 MB for trocr-small).
RUN python -c "\
from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten'); \
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')"

COPY finetune_trocr.py /app/finetune_trocr.py
WORKDIR /app

ENTRYPOINT ["python", "finetune_trocr.py"]
CMD ["--epochs", "2", "--batch-size", "8"]
