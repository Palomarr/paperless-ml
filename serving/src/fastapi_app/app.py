import base64
import io
import math
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# TrOCR – handwritten text recognition
trocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-handwritten", use_fast=False
)
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-small-handwritten"
).to(device)
trocr_model.eval()

# Sentence-Transformers – semantic search
st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
st_model.eval()

from s3_utils import download_image_from_s3

# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "document_chunks"

qdrant_client = None
try:
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HTRRequest(BaseModel):
    document_id: str = "unknown"
    page_id: str = "unknown"
    region_id: str = "unknown"
    crop_s3_url: str = ""
    image_width: int | None = None
    image_height: int | None = None
    image_format: str | None = None
    source: str | None = None
    uploaded_at: str | None = None
    # Fallback: provide base64-encoded image to skip S3 download
    image_base64: str | None = None


class HTRResponse(BaseModel):
    region_id: str
    htr_output: str
    htr_confidence: float
    htr_flagged: bool
    model_version: str
    inference_time_ms: int


class SearchRequest(BaseModel):
    session_id: str = "anonymous"
    query_text: str
    user_id: str = "anonymous"
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: str
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
# Mock document index
# ---------------------------------------------------------------------------

USE_MOCK_CHUNKS = os.environ.get("USE_MOCK_CHUNKS", "true").lower() == "true"

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

if USE_MOCK_CHUNKS:
    chunk_texts: list[str] = [c["text"] for c in MOCK_CHUNKS]
    chunk_embeddings: np.ndarray = st_model.encode(chunk_texts, convert_to_numpy=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HTR_MODEL_VERSION = os.environ.get("HTR_MODEL_VERSION", "htr_v1")
HTR_CONFIDENCE_THRESHOLD = float(os.environ.get("HTR_CONFIDENCE_THRESHOLD", "0.5"))
RETRIEVAL_MODEL_VERSION = os.environ.get("RETRIEVAL_MODEL_VERSION", "retrieval_v1")
SIMILARITY_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Paperless-ngx ML Serving")


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}


# ---------------------------------------------------------------------------
# HTR endpoint
# ---------------------------------------------------------------------------


@app.post("/predict/htr", response_model=HTRResponse)
async def predict_htr(req: HTRRequest) -> HTRResponse:

    # Try image_base64 fallback first, then S3 download
    image: Image.Image | None = None
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid image_base64: {exc}"
            )
    else:
        try:
            image = download_image_from_s3(req.crop_s3_url)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to fetch image from S3 ({req.crop_s3_url}): {exc}. "
                       "Provide image_base64 as fallback.",
            )

    # Inference timing starts after image acquisition
    t0 = time.perf_counter()

    # Preprocess with TrOCR processor
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(
        device
    )

    # Generate with scores for confidence estimation
    with torch.no_grad():
        outputs = trocr_model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode tokens to text
    generated_ids = outputs.sequences
    htr_output = trocr_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Compute sequence-level confidence from token log-probs
    scores = outputs.scores
    log_probs: list[float] = []
    for step_idx, logits in enumerate(scores):
        probs = torch.softmax(logits, dim=-1)
        token_id = generated_ids[0, step_idx + 1]
        token_prob = probs[0, token_id].item()
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


def _search_mock(query_embedding: np.ndarray, top_k: int) -> list[dict]:
    """Search mock in-memory chunks. Returns list of {document_id, chunk_index, chunk_text, similarity_score}."""
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


def _search_qdrant(query_embedding: np.ndarray, top_k: int) -> list[dict]:
    """Search Qdrant, deduplicate to document level (best chunk per doc)."""
    hits = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding.squeeze().tolist(),
        limit=top_k * 3,  # over-fetch to allow dedup
    )

    # Deduplicate: keep best chunk per document_id
    best_per_doc: dict[str, dict] = {}
    for hit in hits:
        doc_id = hit.payload.get("document_id", "")
        score = float(hit.score)
        if doc_id not in best_per_doc or score > best_per_doc[doc_id]["similarity_score"]:
            best_per_doc[doc_id] = {
                "document_id": doc_id,
                "chunk_index": hit.payload.get("chunk_index", 0),
                "chunk_text": hit.payload.get("chunk_text", ""),
                "similarity_score": round(score, 4),
            }

    # Sort by score descending and take top_k
    deduped = sorted(best_per_doc.values(), key=lambda x: x["similarity_score"], reverse=True)
    return deduped[:top_k]


@app.post("/predict/search", response_model=SearchResponse)
async def predict_search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()

    # Encode query
    query_embedding: np.ndarray = st_model.encode([req.query_text], convert_to_numpy=True)

    # Try Qdrant first, fall back to mock chunks, then empty results
    results_raw: list[dict] = []
    fallback_to_keyword = False
    qdrant_failed = False

    if not USE_MOCK_CHUNKS and qdrant_client is not None:
        try:
            results_raw = _search_qdrant(query_embedding, req.top_k)
        except Exception:
            qdrant_failed = True
    elif USE_MOCK_CHUNKS:
        results_raw = _search_mock(query_embedding, req.top_k)
    else:
        qdrant_failed = True

    if qdrant_failed:
        # Qdrant unavailable and not using mock chunks
        fallback_to_keyword = True
        results_raw = []

    # Check similarity threshold
    if results_raw:
        max_score = max(r["similarity_score"] for r in results_raw)
        if max_score < SIMILARITY_THRESHOLD:
            fallback_to_keyword = True

    results = [SearchResult(**r) for r in results_raw]
    inference_time_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        session_id=req.session_id,
        query_text=req.query_text,
        results=results,
        fallback_to_keyword=fallback_to_keyword,
        model_version=RETRIEVAL_MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )
