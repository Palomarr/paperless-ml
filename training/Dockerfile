FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/checkpoints \
    outputs/exported \
    outputs/triton_repo \
    outputs/reports \
    mlruns \
    saved_models

CMD ["python", "train.py"]