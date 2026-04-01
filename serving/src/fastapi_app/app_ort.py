"""FastAPI app using ONNX Runtime instead of PyTorch for inference."""

from __future__ import annotations

import base64
import io
import math
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer, TrOCRProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_DIR = os.environ.get("ONNX_DIR", "/models")
BIENCODER_PATH = os.path.join(ONNX_DIR, "biencoder.onnx")
HTR_DIR = os.path.join(ONNX_DIR, "htr_onnx")

# Pick execution provider based on available hardware
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
active_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in active_providers:
    device = "cuda"
else:
    providers = ["CPUExecutionProvider"]
    device = "cpu"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# MiniLM bi-encoder via ORT
biencoder_session = ort.InferenceSession(BIENCODER_PATH, providers=providers)
biencoder_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# TrOCR via optimum ORT wrapper
from optimum.onnxruntime import ORTModelForVision2Seq

trocr_model = ORTModelForVision2Seq.from_pretrained(HTR_DIR, provider=providers[0])
trocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-handwritten", use_fast=False
)

# ---------------------------------------------------------------------------
# Pydantic models (same schema as PyTorch app)
# ---------------------------------------------------------------------------


class HTRRequest(BaseModel):
    image: str
    document_id: int
    page_number: int
    region_id: int


class HTRResponse(BaseModel):
    document_id: int
    page_number: int
    region_id: int
    transcription: str
    confidence: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: int
    page_number: int
    region_id: int
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    source: str
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Mock document index
# ---------------------------------------------------------------------------

MOCK_CHUNKS: list[dict] = [
    {"text": "The quick brown fox jumps over the lazy dog.", "document_id": 1, "page_number": 1, "region_id": 1},
    {"text": "Machine learning models require large amounts of training data.", "document_id": 1, "page_number": 1, "region_id": 2},
    {"text": "Paperless-ngx is an open-source document management system.", "document_id": 2, "page_number": 1, "region_id": 1},
    {"text": "Optical character recognition converts images of text into machine-readable text.", "document_id": 2, "page_number": 2, "region_id": 1},
    {"text": "Semantic search finds documents by meaning rather than exact keyword match.", "document_id": 3, "page_number": 1, "region_id": 1},
    {"text": "Handwritten text recognition is more challenging than printed text recognition.", "document_id": 3, "page_number": 1, "region_id": 2},
    {"text": "FastAPI is a modern Python web framework for building APIs.", "document_id": 4, "page_number": 1, "region_id": 1},
    {"text": "Docker containers package applications with their dependencies.", "document_id": 4, "page_number": 2, "region_id": 1},
    {"text": "Transformers have revolutionized natural language processing tasks.", "document_id": 5, "page_number": 1, "region_id": 1},
    {"text": "Vector databases store embeddings for efficient similarity search.", "document_id": 5, "page_number": 1, "region_id": 2},
]


def _encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts using the bi-encoder ONNX model with mean pooling."""
    tokens = biencoder_tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)

    outputs = biencoder_session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    # outputs[0] is (batch, seq, 384) — mean pool over sequence dim
    token_embeddings = outputs[0]
    # Mask padding tokens before pooling
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
    return pooled


chunk_texts: list[str] = [c["text"] for c in MOCK_CHUNKS]
chunk_embeddings: np.ndarray = _encode_texts(chunk_texts)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Paperless-ngx ML Serving (ORT)")


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "backend": "onnxruntime"}


# ---------------------------------------------------------------------------
# HTR endpoint
# ---------------------------------------------------------------------------


@app.post("/predict/htr", response_model=HTRResponse)
async def predict_htr(req: HTRRequest) -> HTRResponse:
    image_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values

    generated_ids = trocr_model.generate(
        pixel_values,
        return_dict_in_generate=True,
        output_scores=True,
    )

    transcription = trocr_processor.batch_decode(
        generated_ids.sequences, skip_special_tokens=True
    )[0]

    # Confidence from token log-probs
    scores = generated_ids.scores
    log_probs: list[float] = []
    for step_idx, logits in enumerate(scores):
        probs = np.exp(logits.numpy()) if hasattr(logits, 'numpy') else np.exp(logits)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        token_id = int(generated_ids.sequences[0, step_idx + 1])
        token_prob = float(probs[0, token_id])
        log_probs.append(math.log(max(token_prob, 1e-12)))

    avg_log_prob = sum(log_probs) / max(len(log_probs), 1)
    confidence = round(math.exp(avg_log_prob), 4)

    return HTRResponse(
        document_id=req.document_id,
        page_number=req.page_number,
        region_id=req.region_id,
        transcription=transcription,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


@app.post("/predict/search", response_model=SearchResponse)
async def predict_search(req: SearchRequest) -> SearchResponse:
    query_embedding = _encode_texts([req.query])

    similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = similarities / norms

    ranked_indices = np.argsort(cosine_scores)[::-1]

    top_score = float(cosine_scores[ranked_indices[0]])
    source = "semantic" if top_score >= 0.3 else "keyword"

    results: list[SearchResult] = []
    for idx in ranked_indices[: req.top_k]:
        chunk = MOCK_CHUNKS[int(idx)]
        results.append(
            SearchResult(
                document_id=chunk["document_id"],
                page_number=chunk["page_number"],
                region_id=chunk["region_id"],
                score=round(float(cosine_scores[idx]), 4),
                snippet=chunk["text"],
            )
        )

    return SearchResponse(
        query=req.query,
        source=source,
        results=results,
    )
