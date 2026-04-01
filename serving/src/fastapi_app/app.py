from __future__ import annotations

import base64
import io
import math

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Model loading (module level)
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
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
st_model.eval()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HTRRequest(BaseModel):
    image: str  # base64-encoded image
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
# Mock document index (built at import / startup time)
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

chunk_texts: list[str] = [c["text"] for c in MOCK_CHUNKS]
chunk_embeddings: np.ndarray = st_model.encode(chunk_texts, convert_to_numpy=True)

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
    # Decode base64 image
    image_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
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
    transcription = trocr_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Compute sequence-level confidence from token log-probs
    scores = outputs.scores  # tuple of (vocab_size,) logits per step
    log_probs: list[float] = []
    for step_idx, logits in enumerate(scores):
        probs = torch.softmax(logits, dim=-1)
        # The token chosen at this step (skip BOS at position 0 in sequences)
        token_id = generated_ids[0, step_idx + 1]
        token_prob = probs[0, token_id].item()
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
    # Encode query
    query_embedding: np.ndarray = st_model.encode([req.query], convert_to_numpy=True)

    # Cosine similarity
    similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = similarities / norms

    # Sort descending
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
