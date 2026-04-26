"""FastAPI app using ONNX Runtime instead of PyTorch for inference."""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import time
import uuid

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer, TrOCRProcessor

from s3_utils import (
    download_image_from_s3,
    download_file_from_s3,
    download_prefix_from_s3,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_DIR = os.environ.get("ONNX_DIR", "/models")
BIENCODER_PATH = os.path.join(ONNX_DIR, "biencoder.onnx")
HTR_DIR = os.path.join(ONNX_DIR, "htr_onnx")

# Optional runtime model-fetch URIs. When set, ml-gateway downloads the
# referenced artifacts from MinIO into ONNX_DIR at boot, OVERWRITING the
# image-baked defaults. This is how fine-tuned checkpoints produced by the
# retraining pipeline land in the serving path without rebuilding the
# image — promote_latest() in pipeline-scheduler sets these via compose
# `environment:` on ml-gateway restart.
#
# Expected formats:
#   HTR_MODEL_URI       = s3://paperless-datalake/warehouse/models/trocr-ft-v<N>/onnx/
#                         (prefix containing encoder_model.onnx + decoder_model.onnx + etc.)
#   EMBEDDING_MODEL_URI = s3://paperless-datalake/warehouse/models/mpnet-ft-v<N>/biencoder.onnx
#                         (single file) OR a prefix containing biencoder.onnx.
#
# When unset, the app loads ONNX files that are expected to already exist
# in ONNX_DIR (baked into the image or mounted via compose volume).
HTR_MODEL_URI = os.environ.get("HTR_MODEL_URI", "").strip()
EMBEDDING_MODEL_URI = os.environ.get("EMBEDDING_MODEL_URI", "").strip()

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
# Fetch model artifacts from MinIO if URIs are configured
# ---------------------------------------------------------------------------

if HTR_MODEL_URI:
    logger.warning(
        "HTR_MODEL_URI set (%s) — fetching fine-tuned TrOCR ONNX from MinIO",
        HTR_MODEL_URI,
    )
    try:
        download_prefix_from_s3(HTR_MODEL_URI, HTR_DIR)
        logger.warning("HTR ONNX downloaded to %s", HTR_DIR)
    except RuntimeError as e:
        # Fresh-node fallback: production prefix is empty until first promote.
        # download_prefix_from_s3 raises RuntimeError("no objects found ...")
        # in that case. Fall back to stock HF model so ml-gateway boots cleanly
        # on a fresh stack instead of crash-looping until Airflow promotes.
        if "no objects found" in str(e):
            logger.warning(
                "HTR_MODEL_URI prefix is empty (fresh node, no promote yet) — "
                "falling back to stock microsoft/trocr-small-handwritten",
            )
            HTR_DIR = "microsoft/trocr-small-handwritten"
        else:
            logger.error("HTR_MODEL_URI fetch failed: %s", e)
            raise
    except Exception as e:
        logger.error("HTR_MODEL_URI fetch failed: %s", e)
        raise
else:
    logger.info(
        "HTR_MODEL_URI not set — using image-baked ONNX in %s", HTR_DIR,
    )

if EMBEDDING_MODEL_URI:
    logger.warning(
        "EMBEDDING_MODEL_URI set (%s) — fetching bi-encoder ONNX from MinIO",
        EMBEDDING_MODEL_URI,
    )
    try:
        # Accept either a single-file URI (ends with .onnx) or a directory
        # prefix. Directory form downloads every object under it and we
        # assume one of them is biencoder.onnx at the root.
        if EMBEDDING_MODEL_URI.endswith(".onnx"):
            download_file_from_s3(EMBEDDING_MODEL_URI, BIENCODER_PATH)
        else:
            download_prefix_from_s3(EMBEDDING_MODEL_URI, os.path.dirname(BIENCODER_PATH))
        logger.warning("Bi-encoder ONNX at %s", BIENCODER_PATH)
    except Exception as e:
        logger.error("EMBEDDING_MODEL_URI fetch failed: %s", e)
        raise
else:
    logger.info(
        "EMBEDDING_MODEL_URI not set — using image-baked ONNX at %s",
        BIENCODER_PATH,
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Fresh-node fallback: if the bi-encoder ONNX wasn't pre-baked into the image
# AND EMBEDDING_MODEL_URI didn't fetch one, export from HF at first boot via
# optimum. Mirrors the TrOCR fresh-node path. ~30s one-time on cold boot.
if not os.path.exists(BIENCODER_PATH):
    logger.warning(
        "Bi-encoder ONNX missing at %s — exporting from HF stock "
        "(sentence-transformers/all-mpnet-base-v2). One-time ~30s on first boot.",
        BIENCODER_PATH,
    )
    os.makedirs(ONNX_DIR, exist_ok=True)
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    _tmp = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2",
        export=True,
        provider="CPUExecutionProvider",  # export step doesn't need GPU
    )
    _tmp.save_pretrained(ONNX_DIR)
    # Optimum saves the exported file as model.onnx; rename to our convention.
    src = os.path.join(ONNX_DIR, "model.onnx")
    if os.path.exists(src) and src != BIENCODER_PATH:
        os.rename(src, BIENCODER_PATH)
    del _tmp  # release memory before re-loading via low-level ORT
    logger.warning("Bi-encoder ONNX exported to %s", BIENCODER_PATH)

# mpnet bi-encoder via ORT
biencoder_session = ort.InferenceSession(BIENCODER_PATH, providers=providers)
biencoder_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)

# Verify the ORT session actually selected the GPU provider, not fell back to
# CPU silently. ort.get_available_providers() reports providers built into the
# wheel; session.get_providers() reports what loaded successfully at runtime.
# If they diverge, the device label in the startup banner is a lie.
_actual_providers = biencoder_session.get_providers()
if device == "cuda" and "CUDAExecutionProvider" not in _actual_providers:
    logger.error(
        "device=cuda was claimed but ORT silently fell back to CPU "
        "(actual providers=%s). Inference will run on CPU. "
        "Likely cause: onnxruntime-gpu wheel/CUDA version mismatch.",
        _actual_providers,
    )
    device = "cpu (fallback)"
else:
    logger.warning("ORT bi-encoder providers=%s", _actual_providers)

# TrOCR via optimum ORT wrapper. Supports two input layouts at HTR_DIR:
#
#   (fast path) HTR_DIR contains pre-exported ONNX files:
#     encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx
#     → loads directly, no export step.
#
#   (compat path) HTR_DIR contains an HF safetensors checkpoint only:
#     config.json, model.safetensors, preprocessor_config.json, tokenizer files
#     → optimum runs the ONNX export at load time (~1-3 min cold-boot
#     depending on hardware). Added so we can consume fine-tuned checkpoints
#     from upstream training pipelines that don't emit ONNX themselves
#     (e.g., REDES01/paperless_data_integration/training/trainer.py uses
#     `mlflow.transformers.log_model` which stores safetensors only).
#
# Detection: presence of encoder_model.onnx in HTR_DIR.
from optimum.onnxruntime import ORTModelForVision2Seq

_htr_has_onnx = os.path.exists(os.path.join(HTR_DIR, "encoder_model.onnx"))
if _htr_has_onnx:
    logger.info("HTR dir has pre-exported ONNX — loading directly from %s", HTR_DIR)
    trocr_model = ORTModelForVision2Seq.from_pretrained(
        HTR_DIR, provider=providers[0],
    )
else:
    logger.warning(
        "HTR dir at %s has no encoder_model.onnx — treating as HF checkpoint; "
        "optimum will auto-export to ONNX at load time (~1-3 min)", HTR_DIR,
    )
    trocr_model = ORTModelForVision2Seq.from_pretrained(
        HTR_DIR, provider=providers[0], export=True,
    )
# NOTE: We use TrOCR's AutoProcessor for preprocessing (resize, normalize to
# ImageNet stats, etc.) because our model expects that format.  Elnath's
# feature pipeline preprocesses differently (grayscale, 128px height, [-1,1]
# normalization) but that is for their upstream segmentation — our TrOCR model
# dictates its own preprocessing requirements.
trocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-handwritten", use_fast=False
)

# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

qdrant_client = None
_qdrant_models = None
try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as _qdrant_models

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
except Exception:
    logger.warning("qdrant-client not available; will use mock chunks or fallback")

# Stable namespace for chunk point IDs — matches app.py so Qdrant points
# upserted by either app.py or app_ort.py share the same UUID space.
_CHUNK_ID_NAMESPACE = uuid.UUID("7b93c9e3-ff1f-4c8f-a3bf-ca3f1bf9a0e0")


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def _point_id_for(document_id: str, chunk_idx: int) -> str:
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_idx}"))

# ---------------------------------------------------------------------------
# Pydantic models
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


class EncodeRequest(BaseModel):
    document_id: str
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50


class EncodeResponse(BaseModel):
    document_id: str
    chunks_indexed: int
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
# Mock document index
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
    # outputs[0] is (batch, seq, 768)
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
# Startup banner — pins each deployed run to an explicit model version in the
# container logs. SAFEGUARDING.md §5 accountability mechanism.
# ---------------------------------------------------------------------------
print(
    f"ml-gateway startup: HTR_MODEL_VERSION={HTR_MODEL_VERSION} "
    f"RETRIEVAL_MODEL_VERSION={RETRIEVAL_MODEL_VERSION} "
    f"HTR_CONFIDENCE_THRESHOLD={HTR_CONFIDENCE_THRESHOLD} "
    f"SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD} "
    f"device={device} backend=onnxruntime "
    f"QDRANT={QDRANT_HOST}:{QDRANT_PORT} "
    f"USE_MOCK_CHUNKS={USE_MOCK_CHUNKS} "
    f"HTR_MODEL_URI={HTR_MODEL_URI or '<unset>'}",
    flush=True,
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Paperless-ngx ML Serving (ORT)")


# Prometheus metrics — exposes /metrics with default HTTP histograms,
# plus custom collectors for HTR confidence/corrections and search
# similarity/CTR. The four Counter pairs feed the rollback-trigger alerts
# defined in ops/prometheus/alerts.yml.
try:
    from prometheus_client import Counter, Histogram
    from prometheus_fastapi_instrumentator import Instrumentator

    HTR_CONFIDENCE_HIST = Histogram(
        "htr_confidence",
        "TrOCR per-request confidence (geometric-mean of token probs)",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    )
    SEARCH_SIMILARITY_HIST = Histogram(
        "search_top_similarity",
        "Top-1 cosine similarity score per /search/query call",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    HTR_REQUESTS = Counter(
        "htr_requests_total",
        "Total /htr calls (denominator for correction-rate alert)",
    )
    HTR_CORRECTIONS = Counter(
        "htr_corrections_total",
        "HTR corrections recorded via ml_hooks feedback API",
    )
    SEARCH_QUERIES = Counter(
        "search_queries_total",
        "Total /search/query calls (denominator for CTR alert)",
    )
    SEARCH_CLICKS = Counter(
        "search_clicks_total",
        "Click events on search results reported via ml_hooks feedback API",
    )
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    HTR_CONFIDENCE_HIST = None
    SEARCH_SIMILARITY_HIST = None
    HTR_REQUESTS = None
    HTR_CORRECTIONS = None
    SEARCH_QUERIES = None
    SEARCH_CLICKS = None


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "backend": "onnxruntime"}


# ---------------------------------------------------------------------------
# HTR endpoint
# ---------------------------------------------------------------------------


@app.post("/htr", response_model=HTRResponse)
async def predict_htr(req: HTRRequest) -> HTRResponse:
    if HTR_REQUESTS is not None:
        HTR_REQUESTS.inc()

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

    if HTR_CONFIDENCE_HIST is not None:
        HTR_CONFIDENCE_HIST.observe(htr_confidence)

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


@app.post("/search/query", response_model=SearchResponse)
async def predict_search(req: SearchRequest) -> SearchResponse:
    if SEARCH_QUERIES is not None:
        SEARCH_QUERIES.inc()

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

    # Check similarity threshold + observe top-1 score in Prometheus histogram
    top_score = deduped[0]["similarity_score"] if deduped else 0.0
    if deduped and SEARCH_SIMILARITY_HIST is not None:
        SEARCH_SIMILARITY_HIST.observe(top_score)
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


# ---------------------------------------------------------------------------
# Encode endpoint — index document chunks into Qdrant. Adapts the ORT
# `_encode_texts` helper (mean pooling over the bi-encoder ONNX session)
# in place of sentence-transformers' `.encode()` from app.py.
# ---------------------------------------------------------------------------


@app.post("/search/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest) -> EncodeResponse:
    if qdrant_client is None or _qdrant_models is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")

    t0 = time.perf_counter()

    chunks = _chunk_text(req.text, req.chunk_size, req.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Empty text after stripping")

    embeddings = _encode_texts(chunks)

    # Remove existing chunks for this document so stale entries (from shorter
    # future re-encodes) don't linger. Ignore errors — a fresh collection
    # with no matching points just returns 0.
    try:
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=_qdrant_models.FilterSelector(
                filter=_qdrant_models.Filter(
                    must=[
                        _qdrant_models.FieldCondition(
                            key="document_id",
                            match=_qdrant_models.MatchValue(value=req.document_id),
                        )
                    ]
                )
            ),
        )
    except Exception:
        pass

    points = [
        _qdrant_models.PointStruct(
            id=_point_id_for(req.document_id, idx),
            vector=emb.tolist(),
            payload={
                "document_id": req.document_id,
                "chunk_index": idx,
                "chunk_text": chunks[idx],
                "model_version": RETRIEVAL_MODEL_VERSION,
            },
        )
        for idx, emb in enumerate(embeddings)
    ]

    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    return EncodeResponse(
        document_id=req.document_id,
        chunks_indexed=len(chunks),
        model_version=RETRIEVAL_MODEL_VERSION,
        inference_time_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Internal metric-hook endpoints — fed by ml_hooks fire-and-forget POSTs
# when a user submits an HTR correction or clicks a search result. Bumps
# the rollback-trigger alert numerators (htr_corrections_total,
# search_clicks_total).
# ---------------------------------------------------------------------------


@app.post("/metrics/correction-recorded")
async def correction_recorded():
    if HTR_CORRECTIONS is not None:
        HTR_CORRECTIONS.inc()
    return {"status": "ok"}


@app.post("/metrics/click-recorded")
async def click_recorded():
    if SEARCH_CLICKS is not None:
        SEARCH_CLICKS.inc()
    return {"status": "ok"}
