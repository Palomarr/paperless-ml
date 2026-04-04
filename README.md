# Paperless-ngx ML — Project 02

*Project ID: proj02*
## Overview

This project adds two complementary ML features to [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx), an open-source document management system, deployed on Chameleon Cloud for a hypothetical 30-staff academic department:

1. **Handwritten Text Recognition (HTR)** — TrOCR-small-handwritten transcribes handwritten regions at upload time. Results merge with Tesseract OCR output so handwritten content becomes searchable. Low-confidence transcriptions are flagged for user correction, which feeds back as labeled training data.

2. **Semantic Search** — An all-mpnet-base-v2 bi-encoder (768-dim) encodes documents and queries into a shared embedding space. At query time, the user's search string is encoded and matched against a Qdrant vector index for top-k nearest-neighbor retrieval.


## Repository structure

```
├── contracts/                  # Agreed JSON input/output pairs
│   ├── htr_input.json
│   ├── htr_output.json
│   ├── search_input.json
│   └── search_output.json
├── serving/                    # Serving role (Yikai)
│   ├── dockerfiles/
│   │   ├── Dockerfile.fastapi          # Baseline: FastAPI + PyTorch
│   │   ├── Dockerfile.fastapi-ort      # FastAPI + ONNX Runtime
│   │   └── Dockerfile.dev              # Dev container for exports
│   ├── src/
│   │   ├── fastapi_app/
│   │   │   ├── app.py                  # PyTorch serving app
│   │   │   ├── app_ort.py              # ONNX Runtime serving app
│   │   │   └── s3_utils.py             # MinIO image download helper
│   │   └── export/
│   │       ├── export_onnx.py          # Export both models to ONNX
│   │       └── quantize_onnx.py        # Dynamic quantization (INT8)
│   ├── triton_model_repo/
│   │   ├── htr_model/config.pbtxt
│   │   └── search_model/config.pbtxt
│   ├── benchmarks/
│   │   ├── benchmark_fastapi.py        # Async load test for FastAPI endpoints
│   │   ├── benchmark_triton.sh         # perf_analyzer wrapper for Triton
│   │   ├── baseline_pytorch_cpu.json   # Baseline results (CPU)
│   │   ├── baseline_pytorch_gpu.json   # Baseline results (GPU)
│   │   ├── serving_options.csv         # Summary table of all configurations
│   │   └── triton_notes.md             # Triton design decisions
│   ├── scripts/
│   │   ├── test_contract.py            # Contract validation
│   │   └── upload_test_image.py        # Upload test image to MinIO
│   ├── docker-compose-fastapi.yaml     # Row 1: PyTorch baseline
│   ├── docker-compose-ort.yaml         # Rows 2, 4: ORT FP32
│   ├── docker-compose-ort-quant.yaml   # Row 3: ORT quantized GPU
│   ├── docker-compose-ort-quant-cpu.yaml # Row 5: ORT quantized CPU
│   ├── docker-compose-triton.yaml      # Rows 6–9: Triton
│   ├── setup_serving.sh                # One-command setup from scratch
│   ├── demo_fastapi.sh                 # Demo recording: contract segment
│   └── demo_triton.sh                  # Demo recording: Triton segment
├── training/                   # Training role
├── data/                       # Data role
└── README.md                   # This file
```

## Contracts

The `contracts/` directory contains one JSON input/output pair per model, agreed upon by all three roles. These define the interface between serving, training, and data pipelines:

- **HTR:** Input is a document region (image via S3 URL or base64). Output is transcribed text with a confidence score and a flag for low-confidence results.
- **Search:** Input is a query string with session metadata. Output is ranked document chunks with similarity scores.

## Serving

### Models served

| Model | Purpose | Input | Output | Latency target |
|-------|---------|-------|--------|----------------|
| TrOCR-small-handwritten (encoder) | HTR | Cropped handwriting image (3×384×384) | Encoder hidden states → decoded text | < 5s per page (async) |
| all-mpnet-base-v2 | Semantic search | Query text (tokenized, max 128) | 768-dim embedding → top-k results | < 1s (interactive) |

### Serving configurations evaluated

Nine configurations were benchmarked across three optimization levels, producing the serving options table. (Note: early benchmark rows used all-MiniLM-L6-v2; the production model was later upgraded to all-mpnet-base-v2. Latency characteristics are similar as both are SBERT models of comparable size.)

- **Baseline** (Row 1): FastAPI + PyTorch on GPU. HTR 185ms, Search 20ms.
- **Model-level** (Rows 2–3): ONNX Runtime with FP32 and dynamic quantization. ORT improved HTR by 20%; quantization hurt on GPU due to dequant overhead.
- **Infrastructure-level** (Rows 4–5): CPU-only execution. Sufficient for production load (150 HTR + 50 search requests/day). Marked as cost-priority best option.
- **System-level** (Rows 6–8): Triton Inference Server with ONNX backend. Eliminated Python/FastAPI overhead — HTR latency dropped from 185ms to 17ms (10×), search from 20ms to 7ms. Dynamic batching pushed throughput to 103 rps (HTR) and 198 rps (search with 2 GPU instances).
- **Combined** (Row 9): Quantized ONNX on Triton GPU. Catastrophically worse — dynamic quantization targets CPU ops and forces dequant/requant overhead on GPU. Documented as a negative result.

### Best options by priority

- **Latency:** Triton ONNX baseline (Row 6) — HTR 17ms, Search 7ms
- **Throughput:** Triton + dynamic batching (Row 7 for HTR at 103 rps; Row 8 for Search at 198 rps)
- **Cost:** ORT FP32 on CPU (Row 4) — handles production load without GPU

### Right-sizing

Under sustained Poisson load on the best throughput config (Triton + batching + multi-instance):

| Resource | HTR load | Search load |
|----------|----------|-------------|
| GPU 0 utilization | ~85% | ~76% |
| GPU 0 memory | 1.7 / 16 GB | 1.7 / 16 GB |
| GPU 1 | Idle | Idle |
| CPU | ~120% | ~116% |
| RAM | 2 / 126 GB | 2 / 126 GB |

Single P100 is sufficient. Second P100 and 89% of GPU memory are unused. A smaller GPU (T4, RTX4000) would likely handle production traffic.

### Reproducing from scratch

See `serving/README.md` for full instructions. The shortest path:

1. Run `provision.ipynb` from the Chameleon Jupyter environment (provisions node, installs Docker + NVIDIA toolkit, clones repo).
2. SSH in and run `cd ~/paperless-ml/serving && bash setup_serving.sh`.
3. Both FastAPI and Triton stacks are tested and ready in ~30 minutes.

## Infrastructure

All compute runs on [Chameleon Cloud](https://chameleoncloud.org/) at CHI@TACC.

| Resource | Type | Purpose |
|----------|------|---------|
| GPU node | `gpu_p100` bare metal (2× P100, 48 vCPU, 126 GB) | Model serving and benchmarking |
| OS image | CC-Ubuntu24.04-CUDA | Base image with NVIDIA drivers |
| Triton | nvcr.io/nvidia/tritonserver:24.01-py3 | System-level serving |
| FastAPI base | nvidia/cuda:11.8.0-runtime-ubuntu22.04 | Model-level serving |


## Branching workflow

- `main` — stable shared structure; merge here when features are ready
- `serving` — Yikai's working branch for API endpoints, Dockerfiles, Triton configs, benchmarking
- `training-dev` — Dongting's working branch
- `data-pipeline` — Elnath's working branch

## External datasets

| Dataset | Purpose | License |
|---------|---------|---------|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | HTR pretraining/fine-tuning | Non-commercial research only |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Retrieval model pretraining | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |