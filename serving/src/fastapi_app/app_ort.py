"""FastAPI app using ONNX Runtime instead of PyTorch for inference."""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer, TrOCRProcessor

from s3_utils import download_image_from_s3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_DIR = os.environ.get("ONNX_DIR", "/models")
BIENCODER_PATH = os.path.join(ONNX_DIR, "biencoder.onnx")
HTR_DIR = os.path.join(ONNX_DIR, "htr_onnx")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "document_chunks"

USE_MOCK_CHUNKS = os.environ.get("USE_MOCK_CHUNKS", "true").lower() == "true"
SIMILARITY_THRESHOLD = 0.4

HTR_MODEL_VERSION = os.environ.get("HTR_MODEL_VERSION", "htr_v1")
HTR_CONFIDENCE_THRESHOLD = float(os.environ.get("HTR_CONFIDENCE_THRESHOLD", "0.5"))
RETRIEVAL_MODEL_VERSION = os.environ.get("RETRIEVAL_MODEL_VERSION", "retrieval_v1")

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

# mpnet bi-encoder via ORT
biencoder_session = ort.InferenceSession(BIENCODER_PATH, providers=providers)
biencoder_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)

# TrOCR via optimum ORT wrapper
from optimum.onnxruntime import ORTModelForVision2Seq

trocr_model = ORTModelForVision2Seq.from_pretrained(HTR_DIR, provider=providers[0])
# NOTE: We use TrOCR's AutoProcessor for preprocessing (resize, normalize to
# ImageNet stats, etc.) because our model expects that format.  Elnath's
# feature pipeline preprocesses differently (grayscale, 128px height, [-1,1]
# normalization) but that is for their upstream segmentation — our TrOCR model
# dictates its own preprocessing requirements.
trocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-handwritten", use_fast=False
)

# ---------------------------------------------------------------------------
# Qdrant client (lazy — only used when USE_MOCK_CHUNKS is False)
# ---------------------------------------------------------------------------

qdrant_client = None
try:
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
except Exception:
    logger.warning("qdrant-client not available; will use mock chunks or fallback")

# ---------------------------------------------------------------------------
# Pydantic models — match data-team JSON contracts exactly
# ---------------------------------------------------------------------------


class HTRRequest(BaseModel):
    document_id: str = "unknown"  # UUID
    page_id: str = "unknown"  # UUID
    region_id: str = "unknown"  # UUID
    crop_s3_url: str = ""
    image_width: int | None = None
    image_height: int | None = None
    image_format: str | None = None
    source: str | None = None
    uploaded_at: str | None = None
    # base64 fallback for benchmarking without MinIO
    image_base64: str | None = None


class HTRResponse(BaseModel):
    region_id: str
    htr_output: str
    htr_confidence: float
    htr_flagged: bool
    model_version: str
    inference_time_ms: int


class SearchRequest(BaseModel):
    session_id: str = "anonymous"  # UUID
    query_text: str
    user_id: str = "anonymous"  # UUID
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: str  # UUID
    chunk_index: int
    chunk_text: str
    similarity_score: float


class SearchResponse(BaseModel):
    session_id: str
    query_text: str
    results: list[SearchResult]
    fallback_to_keyword: bool
    model_version: str
    inference_time_ms: int


# ---------------------------------------------------------------------------
# Mock document index (for benchmarking without Qdrant)
# ---------------------------------------------------------------------------

MOCK_CHUNKS: list[dict] = [
    {"text": "The quick brown fox jumps over the lazy dog.", "document_id": "a3f7c2e1-9b4d-4e8a-b5c6-1234567890ab", "chunk_index": 0},
    {"text": "Machine learning models require large amounts of training data.", "document_id": "a3f7c2e1-9b4d-4e8a-b5c6-1234567890ab", "chunk_index": 1},
    {"text": "Paperless-ngx is an open-source document management system.", "document_id": "f1e2d3c4-b5a6-4978-8d6e-5f4a3b2c1d0e", "chunk_index": 0},
    {"text": "Optical character recognition converts images of text into machine-readable text.", "document_id": "f1e2d3c4-b5a6-4978-8d6e-5f4a3b2c1d0e", "chunk_index": 1},
    {"text": "Semantic search finds documents by meaning rather than exact keyword match.", "document_id": "9a8b7c6d-5e4f-4321-0fed-cba987654321", "chunk_index": 0},
    {"text": "Handwritten text recognition is more challenging than printed text recognition.", "document_id": "9a8b7c6d-5e4f-4321-0fed-cba987654321", "chunk_index": 1},
    {"text": "FastAPI is a modern Python web framework for building APIs.", "document_id": "d4e5f6a7-8b9c-4d0e-1f2a-3b4c5d6e7f8a", "chunk_index": 0},
    {"text": "Docker containers package applications with their dependencies.", "document_id": "d4e5f6a7-8b9c-4d0e-1f2a-3b4c5d6e7f8a", "chunk_index": 1},
    {"text": "Transformers have revolutionized natural language processing tasks.", "document_id": "e5f6a7b8-9c0d-4e1f-2a3b-4c5d6e7f8a9b", "chunk_index": 0},
    {"text": "Vector databases store embeddings for efficient similarity search.", "document_id": "e5f6a7b8-9c0d-4e1f-2a3b-4c5d6e7f8a9b", "chunk_index": 1},
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
    # outputs[0] is (batch, seq, 768) — mean pool over sequence dim
    token_embeddings = outputs[0]
    # Mask padding tokens before pooling
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
    return pooled


# Pre-compute mock chunk embeddings for benchmarking mode
if USE_MOCK_CHUNKS:
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
    # Fetch image — try base64 fallback first, then S3
    if req.image_base64:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        try:
            image = download_image_from_s3(req.crop_s3_url)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to fetch image from S3 and no image_base64 fallback provided: {exc}",
            )

    # Start timing after image acquisition (measure inference only)
    t0 = time.perf_counter()

    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values

    generated_ids = trocr_model.generate(
        pixel_values,
        return_dict_in_generate=True,
        output_scores=True,
    )

    htr_output = trocr_processor.batch_decode(
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
    htr_confidence = round(math.exp(avg_log_prob), 4)

    inference_time_ms = int((time.perf_counter() - t0) * 1000)

    return HTRResponse(
        region_id=req.region_id,
        htr_output=htr_output,
        htr_confidence=htr_confidence,
        htr_flagged=htr_confidence < HTR_CONFIDENCE_THRESHOLD,
        model_version=HTR_MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


def _search_qdrant(query_embedding: np.ndarray, top_k: int) -> list[dict] | None:
    """Search Qdrant for similar chunks. Returns None if unavailable."""
    if qdrant_client is None:
        return None
    try:
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding.squeeze().tolist(),
            limit=top_k * 3,  # over-fetch for deduplication
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "document_id": payload.get("document_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "chunk_text": payload.get("chunk_text", ""),
                "similarity_score": float(hit.score),
            })
        return results
    except Exception:
        logger.warning("Qdrant search failed; falling back", exc_info=True)
        return None


def _search_mock(query_embedding: np.ndarray, top_k: int) -> list[dict]:
    """Search mock chunks using cosine similarity."""
    similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = similarities / norms
    ranked_indices = np.argsort(cosine_scores)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        chunk = MOCK_CHUNKS[int(idx)]
        results.append({
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["text"],
            "similarity_score": round(float(cosine_scores[idx]), 4),
        })
    return results


def _deduplicate_to_document(results: list[dict]) -> list[dict]:
    """Keep the best-scoring chunk per document_id."""
    best: dict[str, dict] = {}
    for r in results:
        doc_id = r["document_id"]
        if doc_id not in best or r["similarity_score"] > best[doc_id]["similarity_score"]:
            best[doc_id] = r
    return sorted(best.values(), key=lambda x: x["similarity_score"], reverse=True)


@app.post("/predict/search", response_model=SearchResponse)
async def predict_search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()

    query_embedding = _encode_texts([req.query_text])

    # Try Qdrant first, then mock chunks, then empty fallback
    raw_results: list[dict] | None = None
    fallback_to_keyword = False

    if not USE_MOCK_CHUNKS:
        raw_results = _search_qdrant(query_embedding, req.top_k)

    if raw_results is None and USE_MOCK_CHUNKS:
        raw_results = _search_mock(query_embedding, req.top_k)

    if raw_results is None:
        # Qdrant unavailable and no mock mode — return empty fallback
        inference_time_ms = int((time.perf_counter() - t0) * 1000)
        return SearchResponse(
            session_id=req.session_id,
            query_text=req.query_text,
            results=[],
            fallback_to_keyword=True,
            model_version=RETRIEVAL_MODEL_VERSION,
            inference_time_ms=inference_time_ms,
        )

    # Deduplicate to document level
    deduped = _deduplicate_to_document(raw_results)

    # Check similarity threshold
    top_score = deduped[0]["similarity_score"] if deduped else 0.0
    if top_score < SIMILARITY_THRESHOLD:
        fallback_to_keyword = True

    # Trim to top_k after dedup
    deduped = deduped[: req.top_k]

    results = [
        SearchResult(
            document_id=r["document_id"],
            chunk_index=r["chunk_index"],
            chunk_text=r["chunk_text"],
            similarity_score=r["similarity_score"],
        )
        for r in deduped
    ]

    inference_time_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        session_id=req.session_id,
        query_text=req.query_text,
        results=results,
        fallback_to_keyword=fallback_to_keyword,
        model_version=RETRIEVAL_MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )
