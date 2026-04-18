import time
import uuid

from fastapi import FastAPI

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/htr")
def htr(payload: dict):
    return {
        "region_id": payload.get("region_id", str(uuid.uuid4())),
        "htr_output": "STUB HANDWRITTEN TRANSCRIPTION",
        "htr_confidence": 0.92,
        "htr_flagged": False,
        "model_version": "stub-0.1",
        "inference_time_ms": 5,
    }


@app.post("/search/encode")
def encode(payload: dict):
    return {
        "status": "ok",
        "document_id": payload.get("document_id"),
        "vector_id": str(uuid.uuid4()),
        "model_version": "stub-0.1",
        "indexed_at_ms": int(time.time() * 1000),
    }
