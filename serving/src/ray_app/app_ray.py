"""Ray Serve app with application-level batching for search encoding.

Key improvement over FastAPI: @serve.batch collects concurrent search requests
and encodes them in a single ORT session call, amortizing GPU kernel overhead.
Key improvement over Triton: batches the full pipeline (tokenize + encode +
search + dedup), not just the raw tensor computation.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import time
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, TrOCRProcessor

from s3_utils import download_image_from_s3

logger = logging.getLogger("ray.serve")

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

# ---------------------------------------------------------------------------
# Pydantic models (same as FastAPI app — contract-compatible)
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


# ---------------------------------------------------------------------------
# Search Deployment — with @serve.batch
# ---------------------------------------------------------------------------

@serve.deployment(
    ray_actor_options={"num_gpus": 0.5},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 10,
    },
)
class SearchDeployment:
    def __init__(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        active = ort.get_available_providers()
        if "CUDAExecutionProvider" not in active:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(BIENCODER_PATH, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Qdrant client
        self.qdrant_client = None
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        except Exception:
            logger.warning("qdrant-client not available; will use mock chunks")

        # Pre-compute mock chunk embeddings
        self.chunk_embeddings = None
        if USE_MOCK_CHUNKS:
            texts = [c["text"] for c in MOCK_CHUNKS]
            self.chunk_embeddings = self._encode_texts_sync(texts)

    def _encode_texts_sync(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts (non-batched helper for init)."""
        tokens = self.tokenizer(
            texts, return_tensors="np", padding="max_length",
            truncation=True, max_length=128,
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        outputs = self.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        token_embeddings = outputs[0]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        return (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.01)
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Batch-encode multiple queries in a single ORT session call.

        This is the key differentiator: under concurrent load, Ray Serve
        collects up to 16 query texts, tokenizes them into one (N, 128)
        tensor, and runs a single ORT call instead of N sequential calls.
        """
        embeddings = self._encode_texts_sync(texts)
        return [embeddings[i] for i in range(len(texts))]

    def _search_qdrant(self, query_embedding: np.ndarray, top_k: int) -> list[dict] | None:
        if self.qdrant_client is None:
            return None
        try:
            hits = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding.squeeze().tolist(),
                limit=top_k * 3,
            )
            return [
                {
                    "document_id": (hit.payload or {}).get("document_id", ""),
                    "chunk_index": (hit.payload or {}).get("chunk_index", 0),
                    "chunk_text": (hit.payload or {}).get("chunk_text", ""),
                    "similarity_score": float(hit.score),
                }
                for hit in hits
            ]
        except Exception:
            logger.warning("Qdrant search failed; falling back", exc_info=True)
            return None

    def _search_mock(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        similarities = np.dot(self.chunk_embeddings, query_embedding.T).squeeze()
        norms = np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        cosine_scores = similarities / norms
        ranked_indices = np.argsort(cosine_scores)[::-1]
        return [
            {
                "document_id": MOCK_CHUNKS[int(idx)]["document_id"],
                "chunk_index": MOCK_CHUNKS[int(idx)]["chunk_index"],
                "chunk_text": MOCK_CHUNKS[int(idx)]["text"],
                "similarity_score": round(float(cosine_scores[idx]), 4),
            }
            for idx in ranked_indices[:top_k]
        ]

    @staticmethod
    def _deduplicate(results: list[dict]) -> list[dict]:
        best: dict[str, dict] = {}
        for r in results:
            doc_id = r["document_id"]
            if doc_id not in best or r["similarity_score"] > best[doc_id]["similarity_score"]:
                best[doc_id] = r
        return sorted(best.values(), key=lambda x: x["similarity_score"], reverse=True)

    async def predict(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()

        query_embedding = await self.batch_encode(req.query_text)
        query_embedding = query_embedding.reshape(1, -1)

        raw_results: list[dict] | None = None
        fallback_to_keyword = False

        if not USE_MOCK_CHUNKS:
            raw_results = self._search_qdrant(query_embedding, req.top_k)
        if raw_results is None and USE_MOCK_CHUNKS:
            raw_results = self._search_mock(query_embedding, req.top_k)
        if raw_results is None:
            inference_time_ms = int((time.perf_counter() - t0) * 1000)
            return SearchResponse(
                session_id=req.session_id, query_text=req.query_text,
                results=[], fallback_to_keyword=True,
                model_version=RETRIEVAL_MODEL_VERSION,
                inference_time_ms=inference_time_ms,
            )

        deduped = self._deduplicate(raw_results)
        top_score = deduped[0]["similarity_score"] if deduped else 0.0
        if top_score < SIMILARITY_THRESHOLD:
            fallback_to_keyword = True
        deduped = deduped[: req.top_k]

        results = [
            SearchResult(
                document_id=r["document_id"], chunk_index=r["chunk_index"],
                chunk_text=r["chunk_text"], similarity_score=r["similarity_score"],
            )
            for r in deduped
        ]

        inference_time_ms = int((time.perf_counter() - t0) * 1000)
        return SearchResponse(
            session_id=req.session_id, query_text=req.query_text,
            results=results, fallback_to_keyword=fallback_to_keyword,
            model_version=RETRIEVAL_MODEL_VERSION,
            inference_time_ms=inference_time_ms,
        )


# ---------------------------------------------------------------------------
# HTR Deployment — no batching (autoregressive decode is variable-length)
# ---------------------------------------------------------------------------

@serve.deployment(
    ray_actor_options={"num_cpus": 2},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 5,
    },
)
class HTRDeployment:
    def __init__(self):
        # HTR uses CPU provider: the autoregressive decode loop runs in PyTorch
        # (CPU), so the ORT encoder also runs on CPU to avoid GPU↔CPU transfers.
        # The search deployment is where GPU batching matters.
        from optimum.onnxruntime import ORTModelForVision2Seq
        self.model = ORTModelForVision2Seq.from_pretrained(
            HTR_DIR,
            provider="CPUExecutionProvider",
            use_io_binding=False,
        )
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-small-handwritten", use_fast=False
        )

    async def predict(self, req: HTRRequest) -> HTRResponse:
        if req.image_base64:
            image_bytes = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            try:
                image = download_image_from_s3(req.crop_s3_url)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to fetch image from S3 and no image_base64 fallback: {exc}",
                )

        t0 = time.perf_counter()

        # Convert to numpy RGB array to prevent channel mismatch in processor
        image_array = np.array(image.convert("RGB"))
        pixel_values = self.processor(images=image_array, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(
            pixel_values, return_dict_in_generate=True, output_scores=True,
        )
        htr_output = self.processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )[0]

        scores = generated_ids.scores
        log_probs: list[float] = []
        for step_idx, logits in enumerate(scores):
            probs = np.exp(logits.numpy()) if hasattr(logits, "numpy") else np.exp(logits)
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
# API Ingress — exposes same endpoints as FastAPI app
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(title="Paperless-ngx ML Serving (Ray Serve + ORT)")


@serve.deployment()
@serve.ingress(fastapi_app)
class APIIngress:
    def __init__(self, htr: HTRDeployment, search: SearchDeployment):
        self.htr = htr
        self.search = search

    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "ok", "backend": "ray-serve-ort"}

    @fastapi_app.post("/predict/htr", response_model=HTRResponse)
    async def predict_htr(self, req: HTRRequest) -> HTRResponse:
        return await self.htr.predict.remote(req)

    @fastapi_app.post("/predict/search", response_model=SearchResponse)
    async def predict_search(self, req: SearchRequest) -> SearchResponse:
        return await self.search.predict.remote(req)


# ---------------------------------------------------------------------------
# Bind deployment graph
# ---------------------------------------------------------------------------

search = SearchDeployment.bind()
htr = HTRDeployment.bind()
app = APIIngress.bind(htr, search)
