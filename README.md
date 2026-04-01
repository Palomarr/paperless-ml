# Paperless-ngx: Handwriting Recognition & Semantic Search

**ECE-GY 9183 — ML Systems Engineering & Operations**

Team: Dongting Gao (Training), Yikai Sun (Serving), Elnath Zhao (Data)

Project ID: proj02

## Overview

This project adds two complementary ML features to [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx), an open-source document management system, deployed on Chameleon Cloud for a hypothetical 30-staff academic department:

1. **Handwriting Text Recognition (HTR)** — transcribes handwritten regions at upload time and merges results with Tesseract OCR output.
2. **Semantic Search** — a bi-encoder retrieval model that encodes documents and queries into a shared embedding space for nearest-neighbor search.

## Repository Structure

```
├── contracts/          # Shared JSON input/output pairs
│   ├── htr_input.json
│   ├── htr_output.json
│   ├── search_input.json
│   └── search_output.json
├── serving/            # Serving role (Yikai)
│   ├── dockerfiles/
│   ├── src/
│   └── benchmarks/
├── training/           # Training role (Dongting)
├── data/               # Data role (Elnath)
└── README.md
```

## Branching Workflow

- `main` — stable, shared structure. Merge here when features are ready.
- `serving` — Yikai's working branch for API endpoints, Dockerfiles, Triton configs, benchmarking.
- `training-dev` — Dongting's working branch.
- `data-pipeline` — Elnath's working branch.

## Contracts

The `contracts/` directory contains one JSON input/output pair per model. All three roles must agree on these schemas before coding against them. See the files for current field definitions.

## Infrastructure

All compute runs on [Chameleon Cloud](https://chameleoncloud.org/). Typical resources for serving:

- **GPU node:** `gpu_p100` bare metal at CHI@TACC (2× NVIDIA P100)
- **Image:** `CC-Ubuntu24.04-CUDA`

### Naming convention

All Chameleon resources (leases, instances, volumes, etc.) must end with `proj02`.

## Quick Start (Serving)

1. **Reserve** a `gpu_p100` node on Chameleon (see Host Calendar at CHI@TACC).
2. **Provision** using the notebook: open `serving/provision_serving.ipynb` in the Chameleon Jupyter environment.
3. **SSH** into the instance: `ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING_IP>`
4. **Build and run** the serving container:
   ```bash
   cd <repo>/serving
   docker compose -f docker-compose-fastapi.yaml build
   docker compose -f docker-compose-fastapi.yaml up -d
   ```
5. **Test** the API at `http://<FLOATING_IP>:8000/docs`

## External Datasets

| Dataset | Purpose | License |
|---|---|---|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | HTR pretraining | Non-commercial research only |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Retrieval model pretraining | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
